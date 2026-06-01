#!/usr/bin/env python
"""Attribute scenario variation to starting level versus narrative.

This script consumes saved fixed-start rollout bundles and runs a crossed
two-factor analysis:

    starting level x professional narrative

The goal is not to claim causal identification. It is a product diagnostic that
answers whether the narrative has measurable impact after controlling for the
accepted starting level, and how large that impact is relative to start-level
variation and start/narrative interaction.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_fixed_start_rollout_policy_comparison import (  # noqa: E402
    CASE_NAMES,
    FACTOR_INDEX,
    SUMMARY_FACTORS,
    _case_arrays_path,
    _case_report_path,
    _direction_status,
    _ks_statistic,
    _load_json,
    _operational_variant_index,
    _portfolio_pnl,
    _portfolio_stats,
    _support_jaccard,
    _support_rows,
    _terminal_summary,
    normalize_case_name,
    public_case_label,
)
from experiments.backfill.block_ar.nl_component_pooling_diagnostic import (  # noqa: E402
    component_slices_for_variant,
)
from experiments.backfill.block_ar.nl_support_component_posterior_bakeoff import (  # noqa: E402
    POSTERIOR_MODES,
    PLOT_POSTERIOR_LABELS,
)
from experiments.backfill.block_ar.nl_sparse_component_family_view import (  # noqa: E402
    select_sparse_components,
)


DEFAULT_START_ROOTS = {
    "start18": (
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "posterior_ensemble_candidate_963a_start18_s384_d400"
    ),
    "start22": (
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "posterior_ensemble_candidate_962a_start22_s384_d400"
    ),
    "start40": (
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "posterior_ensemble_candidate_963b_start40_s384_d400"
    ),
}
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "posterior_ensemble_candidate_964a_start_narrative_attribution_top3_90"
)
DEFAULT_PAPER_TABLE = (
    "paper/narrative_grounded_scenarios/generated_tables/"
    "table_start_narrative_attribution.tex"
)
DEFAULT_PAPER_FIGURE = (
    "paper/narrative_grounded_scenarios/figures/" "start_narrative_attribution.png"
)


@dataclass(frozen=True)
class AttributionResult:
    feature_space: str
    policy: str
    start_share: float
    narrative_share: float
    interaction_share: float
    total_ss: float
    mean_feature_distance_same_start_narrative: float
    mean_feature_distance_same_narrative_start: float
    mean_portfolio_ks_same_start_narrative: float
    mean_portfolio_ks_same_narrative_start: float
    mean_factor_ks_same_start_narrative: float
    mean_factor_ks_same_narrative_start: float
    mean_support_jaccard_same_start_narrative: float
    mean_support_jaccard_same_narrative_start: float


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text(path: str | Path, text: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text.rstrip() + "\n")


def _parse_start_root(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError("--start-root must be LABEL=PATH")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError("--start-root must be LABEL=PATH")
    return label, path


def _terminal_stats(values: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    q10, q50, q90 = np.percentile(arr, [10, 50, 90])
    return [float(np.mean(arr)), float(np.std(arr)), float(q10), float(q50), float(q90)]


def cell_feature_vector(
    result: dict[str, Any],
    *,
    feature_space: str,
) -> tuple[np.ndarray, list[str]]:
    """Return one distribution summary vector for a start/narrative cell."""

    if feature_space not in {"raw_level", "start_normalized"}:
        raise ValueError(f"unknown feature_space {feature_space!r}")
    states = np.asarray(result["states"], dtype=np.float64)
    start = np.asarray(result["start"], dtype=np.float64).reshape(-1)
    values: list[float] = []
    names: list[str] = []
    for factor in SUMMARY_FACTORS:
        idx = FACTOR_INDEX[factor]
        terminal = states[:, -1, idx]
        if feature_space == "start_normalized":
            scale = max(abs(float(start[idx])), 1.0)
            terminal = (terminal - float(start[idx])) / scale
        for stat_name, stat_value in zip(
            ("mean", "std", "p10", "p50", "p90"),
            _terminal_stats(terminal),
            strict=True,
        ):
            names.append(f"{factor}_{stat_name}")
            values.append(stat_value)

    pnl_terminal = _portfolio_pnl(states, start)[:, -1]
    q05 = float(np.percentile(pnl_terminal, 5))
    portfolio_stats = {
        "p10": float(np.percentile(pnl_terminal, 10)),
        "p50": float(np.percentile(pnl_terminal, 50)),
        "p90": float(np.percentile(pnl_terminal, 90)),
        "std": float(np.std(pnl_terminal)),
        "var95_loss": float(-q05),
        "es95_loss": float(-np.mean(pnl_terminal[pnl_terminal <= q05])),
    }
    for key, value in portfolio_stats.items():
        names.append(f"portfolio_{key}")
        values.append(float(value))
    return np.asarray(values, dtype=np.float64), names


def _standardize_cube(cube: np.ndarray) -> np.ndarray:
    flat = cube.reshape(-1, cube.shape[-1])
    scale = np.std(flat, axis=0)
    scale[scale < 1.0e-10] = 1.0
    return (cube - np.mean(flat, axis=0)) / scale


def two_way_feature_attribution(cube: np.ndarray) -> dict[str, float]:
    """Two-way ANOVA-style sums of squares for a complete start x narrative cube."""

    x = _standardize_cube(np.asarray(cube, dtype=np.float64))
    n_start, n_narrative, _ = x.shape
    grand = np.mean(x, axis=(0, 1))
    start_mean = np.mean(x, axis=1)
    narrative_mean = np.mean(x, axis=0)
    total_ss = float(np.sum((x - grand) ** 2))
    start_ss = float(n_narrative * np.sum((start_mean - grand) ** 2))
    narrative_ss = float(n_start * np.sum((narrative_mean - grand) ** 2))
    interaction = x - start_mean[:, None, :] - narrative_mean[None, :, :] + grand
    interaction_ss = float(np.sum(interaction**2))
    denom = total_ss if total_ss > 0 else 1.0
    return {
        "total_ss": total_ss,
        "start_ss": start_ss,
        "narrative_ss": narrative_ss,
        "interaction_ss": interaction_ss,
        "start_share": start_ss / denom,
        "narrative_share": narrative_ss / denom,
        "interaction_share": interaction_ss / denom,
    }


def _pairwise_feature_distances(cube: np.ndarray) -> dict[str, float]:
    x = _standardize_cube(np.asarray(cube, dtype=np.float64))
    n_start, n_narrative, _ = x.shape
    narrative_dists = []
    for s in range(n_start):
        for i in range(n_narrative):
            for j in range(i + 1, n_narrative):
                narrative_dists.append(float(np.linalg.norm(x[s, i] - x[s, j])))
    start_dists = []
    for n in range(n_narrative):
        for i in range(n_start):
            for j in range(i + 1, n_start):
                start_dists.append(float(np.linalg.norm(x[i, n] - x[j, n])))
    return {
        "same_start_narrative": (
            float(np.mean(narrative_dists)) if narrative_dists else 0.0
        ),
        "same_narrative_start": float(np.mean(start_dists)) if start_dists else 0.0,
    }


def _pairwise_distribution_metrics(
    grid: dict[str, dict[str, dict[str, Any]]],
    *,
    starts: tuple[str, ...],
    cases: tuple[str, ...],
) -> dict[str, float]:
    narrative_portfolio_ks = []
    narrative_factor_ks = []
    narrative_support_jaccard = []
    for start in starts:
        for i, left_case in enumerate(cases):
            for right_case in cases[i + 1 :]:
                left = grid[start][left_case]
                right = grid[start][right_case]
                left_pnl = _portfolio_pnl(left["states"], left["start"])[:, -1]
                right_pnl = _portfolio_pnl(right["states"], right["start"])[:, -1]
                narrative_portfolio_ks.append(_ks_statistic(left_pnl, right_pnl))
                factor_ks = []
                for factor in SUMMARY_FACTORS:
                    idx = FACTOR_INDEX[factor]
                    factor_ks.append(
                        _ks_statistic(
                            np.asarray(left["states"])[:, -1, idx],
                            np.asarray(right["states"])[:, -1, idx],
                        )
                    )
                narrative_factor_ks.append(float(np.mean(factor_ks)))
                narrative_support_jaccard.append(
                    _support_jaccard(left["support"], right["support"])
                )

    start_portfolio_ks = []
    start_factor_ks = []
    start_support_jaccard = []
    for case in cases:
        for i, left_start in enumerate(starts):
            for right_start in starts[i + 1 :]:
                left = grid[left_start][case]
                right = grid[right_start][case]
                left_pnl = _portfolio_pnl(left["states"], left["start"])[:, -1]
                right_pnl = _portfolio_pnl(right["states"], right["start"])[:, -1]
                start_portfolio_ks.append(_ks_statistic(left_pnl, right_pnl))
                factor_ks = []
                for factor in SUMMARY_FACTORS:
                    idx = FACTOR_INDEX[factor]
                    factor_ks.append(
                        _ks_statistic(
                            np.asarray(left["states"])[:, -1, idx],
                            np.asarray(right["states"])[:, -1, idx],
                        )
                    )
                start_factor_ks.append(float(np.mean(factor_ks)))
                start_support_jaccard.append(
                    _support_jaccard(left["support"], right["support"])
                )

    return {
        "mean_portfolio_ks_same_start_narrative": float(
            np.mean(narrative_portfolio_ks)
        ),
        "mean_portfolio_ks_same_narrative_start": float(np.mean(start_portfolio_ks)),
        "mean_factor_ks_same_start_narrative": float(np.mean(narrative_factor_ks)),
        "mean_factor_ks_same_narrative_start": float(np.mean(start_factor_ks)),
        "mean_support_jaccard_same_start_narrative": float(
            np.mean(narrative_support_jaccard)
        ),
        "mean_support_jaccard_same_narrative_start": float(
            np.mean(start_support_jaccard)
        ),
    }


def load_grid(
    *,
    start_roots: dict[str, str],
    cases: tuple[str, ...],
    policy: str,
    posterior_mode: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    grid: dict[str, dict[str, dict[str, Any]]] = {}
    for start, root in start_roots.items():
        grid[start] = {}
        for case in cases:
            grid[start][case] = load_case_result(
                output_root=root,
                case_name=case,
                policy_name=policy,
                posterior_mode=posterior_mode,
            )
    return grid


def analyze_policy(
    *,
    start_roots: dict[str, str],
    cases: tuple[str, ...],
    policy: str,
    feature_space: str,
    posterior_mode: str,
) -> tuple[AttributionResult, dict[str, Any]]:
    starts = tuple(start_roots)
    grid = load_grid(
        start_roots=start_roots,
        cases=cases,
        policy=policy,
        posterior_mode=posterior_mode,
    )
    vectors = []
    feature_names: list[str] | None = None
    for start in starts:
        row = []
        for case in cases:
            vector, names = cell_feature_vector(
                grid[start][case], feature_space=feature_space
            )
            feature_names = names if feature_names is None else feature_names
            row.append(vector)
        vectors.append(row)
    cube = np.asarray(vectors, dtype=np.float64)
    decomp = two_way_feature_attribution(cube)
    feature_dist = _pairwise_feature_distances(cube)
    dist_metrics = _pairwise_distribution_metrics(grid, starts=starts, cases=cases)
    result = AttributionResult(
        feature_space=feature_space,
        policy=policy,
        start_share=float(decomp["start_share"]),
        narrative_share=float(decomp["narrative_share"]),
        interaction_share=float(decomp["interaction_share"]),
        total_ss=float(decomp["total_ss"]),
        mean_feature_distance_same_start_narrative=float(
            feature_dist["same_start_narrative"]
        ),
        mean_feature_distance_same_narrative_start=float(
            feature_dist["same_narrative_start"]
        ),
        mean_portfolio_ks_same_start_narrative=float(
            dist_metrics["mean_portfolio_ks_same_start_narrative"]
        ),
        mean_portfolio_ks_same_narrative_start=float(
            dist_metrics["mean_portfolio_ks_same_narrative_start"]
        ),
        mean_factor_ks_same_start_narrative=float(
            dist_metrics["mean_factor_ks_same_start_narrative"]
        ),
        mean_factor_ks_same_narrative_start=float(
            dist_metrics["mean_factor_ks_same_narrative_start"]
        ),
        mean_support_jaccard_same_start_narrative=float(
            dist_metrics["mean_support_jaccard_same_start_narrative"]
        ),
        mean_support_jaccard_same_narrative_start=float(
            dist_metrics["mean_support_jaccard_same_narrative_start"]
        ),
    )
    details = {
        "starts": list(starts),
        "cases": list(cases),
        "feature_names": feature_names or [],
        "decomposition": decomp,
        "pairwise_feature_distances": feature_dist,
        "pairwise_distribution_metrics": dist_metrics,
    }
    return result, details


def _result_to_dict(result: AttributionResult) -> dict[str, Any]:
    return {key: value for key, value in result.__dict__.items()}


def _format_pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}\\%"


def render_latex_table(results: list[AttributionResult]) -> str:
    rows = []
    for result in results:
        if result.policy == "start_only_topk" and result.feature_space == "raw_level":
            continue
        label = {
            "current_start_checked_gap30": "Narrative-conditioned",
            "cohesive_support_gap30": "Nearest-similar top3/90",
            "start_only_topk": "Start-only null",
        }.get(result.policy, result.policy.replace("_", " "))
        space = {
            "raw_level": "Raw terminal levels",
            "start_normalized": "Start-normalized terminal moves",
        }.get(result.feature_space, result.feature_space)
        rows.append(
            " & ".join(
                [
                    label,
                    space,
                    _format_pct(result.start_share),
                    _format_pct(result.narrative_share),
                    _format_pct(result.interaction_share),
                    f"{result.mean_factor_ks_same_start_narrative:.3f}",
                    f"{result.mean_factor_ks_same_narrative_start:.3f}",
                    f"{result.mean_portfolio_ks_same_start_narrative:.3f}",
                    f"{result.mean_portfolio_ks_same_narrative_start:.3f}",
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Start-versus-narrative attribution for the promoted nearest-similar top-3/90 posterior view across accepted starts and six professional narratives. Shares are computed from a crossed ANOVA-style decomposition of standardized distribution-summary features. Same-start columns compare narratives after holding the starting level fixed; same-narrative columns compare starting levels after holding the narrative fixed.}",
            r"\label{tab:start_narrative_attribution}",
            r"\scriptsize",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{llrrrrrrr}",
            r"\toprule",
            r"Policy & Feature space & Start share & Narrative share & Interaction & Factor KS same start & Factor KS same narrative & Portfolio KS same start & Portfolio KS same narrative \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Start-versus-Narrative Attribution",
        "",
        "This crossed diagnostic estimates how much scenario-distribution variation is associated with the accepted starting level, the professional narrative, and their interaction.",
        "",
        f"Posterior view: `{payload['posterior_mode']}` ({payload['posterior_label']}).",
        "",
        "Method: each start/narrative cell is summarized by standardized terminal factor and portfolio-risk features, then decomposed with a two-factor ANOVA-style sum-of-squares calculation. Pairwise KS columns are computed directly from generated distributions.",
        "",
        "| Policy | Feature space | Start share | Narrative share | Interaction | Factor KS same start | Factor KS same narrative | Portfolio KS same start | Portfolio KS same narrative |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["results"]:
        policy_label = {
            "current_start_checked_gap30": "Narrative-conditioned",
            "cohesive_support_gap30": "Nearest-similar top3/90",
            "start_only_topk": "Start-only null",
        }.get(row["policy"], row["policy"])
        lines.append(
            "| {policy} | {space} | {start:.1%} | {narr:.1%} | {inter:.1%} | "
            "{factor_n:.3f} | {factor_s:.3f} | {port_n:.3f} | {port_s:.3f} |".format(
                policy=policy_label,
                space=row["feature_space"],
                start=row["start_share"],
                narr=row["narrative_share"],
                inter=row["interaction_share"],
                factor_n=row["mean_factor_ks_same_start_narrative"],
                factor_s=row["mean_factor_ks_same_narrative_start"],
                port_n=row["mean_portfolio_ks_same_start_narrative"],
                port_s=row["mean_portfolio_ks_same_narrative_start"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Raw-level features intentionally include the level geometry a risk manager sees on a chart, so the starting level is expected to explain a large share.",
            "- Start-normalized features ask the cleaner product question: after anchoring each path at its accepted start, how much variation is still attributable to narrative and start/narrative interaction?",
            "- The start-only null should have zero or near-zero same-start narrative KS; otherwise the attribution design is broken.",
        ]
    )
    return "\n".join(lines)


def plot_attribution_bars(
    *,
    results: list[AttributionResult],
    output_path: str | Path,
    posterior_label: str,
) -> None:
    import matplotlib.pyplot as plt

    plot_rows = [
        result
        for result in results
        if not (
            result.policy == "start_only_topk" and result.feature_space == "raw_level"
        )
    ]
    labels = [
        (
            "Narrative\nraw"
            if r.policy != "start_only_topk" and r.feature_space == "raw_level"
            else (
                "Narrative\nnormalized"
                if r.policy != "start_only_topk"
                else "Start-only\nnormalized"
            )
        )
        for r in plot_rows
    ]
    start = np.asarray([r.start_share for r in plot_rows])
    narrative = np.asarray([r.narrative_share for r in plot_rows])
    interaction = np.asarray([r.interaction_share for r in plot_rows])
    x = np.arange(len(plot_rows))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x, start, label="Starting level", color="#5E81AC")
    ax.bar(x, narrative, bottom=start, label="Narrative", color="#A3BE8C")
    ax.bar(
        x,
        interaction,
        bottom=start + narrative,
        label="Start x narrative",
        color="#EBCB8B",
    )
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share of standardized feature variation")
    ax.set_title(f"Start-versus-narrative attribution: {posterior_label}")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _posterior_sample_indices(
    *,
    arrays: dict[str, Any],
    variant_index: int,
    sample_count: int,
    posterior_mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return sample indices and selected support components for a posterior view."""

    if posterior_mode not in POSTERIOR_MODES:
        raise ValueError(
            f"unknown posterior_mode {posterior_mode!r}; "
            f"expected one of {sorted(POSTERIOR_MODES)}"
        )
    components = component_slices_for_variant(
        variant_index=int(variant_index),
        component_variant_index=np.asarray(
            arrays["rollout_component_variant_index"], dtype=np.int64
        ),
        component_window_index=np.asarray(
            arrays["rollout_component_window_index"], dtype=np.int64
        ),
        component_weight=np.asarray(
            arrays["rollout_component_weight"], dtype=np.float64
        ),
        component_sample_count=np.asarray(
            arrays["rollout_component_sample_count"], dtype=np.int64
        ),
        sample_count=int(sample_count),
    )
    components = [
        {**component, "component_no": int(pos)}
        for pos, component in enumerate(components)
    ]
    if posterior_mode == "full":
        selected = [dict(component) for component in components]
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


def _selected_support_rows(
    *,
    support: list[dict[str, Any]],
    selected_components: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_idx = {int(row["window_index"]): dict(row) for row in support}
    total = sum(max(float(row.get("weight", 0.0)), 0.0) for row in selected_components)
    rows: list[dict[str, Any]] = []
    for rank, component in enumerate(selected_components, start=1):
        idx = int(component["window_index"])
        row = by_idx.get(idx, {"window_index": idx, "window_id": ""})
        weight = max(float(component.get("weight", 0.0)), 0.0)
        sparse_weight = float(component.get("sparse_weight", 0.0) or 0.0)
        if sparse_weight <= 0.0:
            sparse_weight = (
                weight / total
                if total > 0.0
                else 1.0 / max(len(selected_components), 1)
            )
        rows.append(
            {
                **row,
                "rank": int(rank),
                "weight": float(sparse_weight),
                "base_weight": float(weight),
                "sample_count": int(component.get("sample_count", 0) or 0),
            }
        )
    return rows


def load_case_result(
    *,
    output_root: str | Path,
    case_name: str,
    policy_name: str,
    posterior_mode: str,
) -> dict[str, Any]:
    report_path = _case_report_path(output_root, case_name, policy_name)
    arrays_path = _case_arrays_path(output_root, case_name, policy_name)
    report = _load_json(report_path)
    arrays = dict(np.load(arrays_path))
    op_idx = _operational_variant_index(report, arrays)
    all_states = np.asarray(arrays["generated_states"], dtype=np.float32)[op_idx]
    sample_indices, selected_components = _posterior_sample_indices(
        arrays=arrays,
        variant_index=op_idx,
        sample_count=int(all_states.shape[0]),
        posterior_mode=posterior_mode,
    )
    states = all_states[sample_indices]
    start = np.asarray(arrays["requested_raw"], dtype=np.float32)[op_idx]
    support = _selected_support_rows(
        support=_support_rows(report),
        selected_components=selected_components,
    )
    return {
        "case": normalize_case_name(case_name),
        "case_label": public_case_label(case_name),
        "policy": str(policy_name),
        "posterior_mode": str(posterior_mode),
        "report_path": str(report_path),
        "arrays_path": str(arrays_path),
        "generated_shape": [int(v) for v in states.shape],
        "start": start,
        "states": states,
        "support": support,
        "support_count": int(len(support)),
        "direction_status": _direction_status(report),
        "terminal_raw_levels": _terminal_summary(
            states, {name: FACTOR_INDEX[name] for name in SUMMARY_FACTORS}
        ),
        "portfolio_stats": _portfolio_stats(states, start),
    }


def run_attribution(args: argparse.Namespace) -> dict[str, Any]:
    start_roots = dict(DEFAULT_START_ROOTS)
    if args.start_root:
        start_roots = dict(_parse_start_root(item) for item in args.start_root)
    cases = tuple(normalize_case_name(case) for case in (args.case or CASE_NAMES))
    policies = tuple(args.policy or ["current_start_checked_gap30", "start_only_topk"])
    posterior_mode = str(args.posterior_mode)
    if posterior_mode not in POSTERIOR_MODES:
        raise ValueError(
            f"unknown --posterior-mode {posterior_mode!r}; "
            f"expected one of {sorted(POSTERIOR_MODES)}"
        )
    output_dir = Path(args.output_dir)

    results: list[AttributionResult] = []
    details: dict[str, Any] = {}
    for policy in policies:
        details[policy] = {}
        for feature_space in ("raw_level", "start_normalized"):
            result, detail = analyze_policy(
                start_roots=start_roots,
                cases=cases,
                policy=policy,
                feature_space=feature_space,
                posterior_mode=posterior_mode,
            )
            results.append(result)
            details[policy][feature_space] = detail

    payload = {
        "status": "ok",
        "scope": "crossed start x narrative attribution over saved rollout bundles",
        "start_roots": start_roots,
        "cases": list(cases),
        "case_labels": {case: public_case_label(case) for case in cases},
        "policies": list(policies),
        "posterior_mode": posterior_mode,
        "posterior_label": str(
            PLOT_POSTERIOR_LABELS.get(posterior_mode, posterior_mode)
        ),
        "results": [_result_to_dict(result) for result in results],
        "details": details,
        "related_work_note": (
            "The analysis follows variance-based sensitivity / functional "
            "ANOVA practice: decompose output variation into main effects and "
            "interaction terms, and use common rollout settings to reduce "
            "stochastic comparison noise."
        ),
    }
    json_path = output_dir / "start_narrative_attribution.json"
    md_path = output_dir / "start_narrative_attribution.md"
    figure_path = output_dir / "start_narrative_attribution.png"
    _write_json(json_path, payload)
    _write_text(md_path, render_markdown(payload))
    plot_attribution_bars(
        results=results,
        output_path=figure_path,
        posterior_label=str(PLOT_POSTERIOR_LABELS.get(posterior_mode, posterior_mode)),
    )
    payload["artifact_paths"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "figure": str(figure_path),
    }
    _write_json(json_path, payload)

    if not args.no_paper:
        table_path = Path(args.paper_table)
        figure_paper_path = Path(args.paper_figure)
        _write_text(table_path, render_latex_table(results))
        figure_paper_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(figure_path, figure_paper_path)
        payload["paper_artifacts"] = {
            "table": str(table_path),
            "figure": str(figure_paper_path),
        }
        _write_json(json_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-root", action="append", help="LABEL=rollout-root")
    parser.add_argument("--case", action="append")
    parser.add_argument("--policy", action="append")
    parser.add_argument(
        "--posterior-mode",
        default="top3_90",
        choices=tuple(POSTERIOR_MODES),
        help="Posterior component view to apply before attribution.",
    )
    parser.add_argument("--no-paper", action="store_true")
    parser.add_argument("--paper-table", default=DEFAULT_PAPER_TABLE)
    parser.add_argument("--paper-figure", default=DEFAULT_PAPER_FIGURE)
    args = parser.parse_args()
    payload = run_attribution(args)
    compact = {
        "status": payload["status"],
        "posterior_mode": payload["posterior_mode"],
        "posterior_label": payload["posterior_label"],
        "artifact_paths": payload["artifact_paths"],
        "paper_artifacts": payload.get("paper_artifacts", {}),
        "results": [
            {
                "policy": row["policy"],
                "feature_space": row["feature_space"],
                "start_share": row["start_share"],
                "narrative_share": row["narrative_share"],
                "interaction_share": row["interaction_share"],
                "factor_ks_same_start": row["mean_factor_ks_same_start_narrative"],
                "factor_ks_same_narrative": row["mean_factor_ks_same_narrative_start"],
                "portfolio_ks_same_start": row[
                    "mean_portfolio_ks_same_start_narrative"
                ],
                "portfolio_ks_same_narrative": row[
                    "mean_portfolio_ks_same_narrative_start"
                ],
            }
            for row in payload["results"]
        ],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
