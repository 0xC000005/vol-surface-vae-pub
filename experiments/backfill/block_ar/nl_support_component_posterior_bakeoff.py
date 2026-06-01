#!/usr/bin/env python
"""Compare component-posterior readouts for fixed-start support policies.

This artifact-only analyzer consumes outputs from
``nl_fixed_start_rollout_policy_comparison.py``. It does not call OpenAI,
train a model, or rerun the frozen SNI generator. The goal is to measure
whether narrative signal is being lost when support components are fully pooled.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_component_pooling_diagnostic import (  # noqa: E402
    component_slices_for_variant,
)
from experiments.backfill.block_ar.nl_fixed_start_rollout_policy_comparison import (  # noqa: E402
    CASE_NAMES,
    CASE_SPECS,
    FACTOR_INDEX,
    POLICIES,
    SUMMARY_FACTORS,
    _case_report_path,
    _ks_statistic,
    _portfolio_pnl,
    public_case_label,
)
from experiments.backfill.block_ar.nl_sparse_component_family_view import (  # noqa: E402
    _variant_index,
    select_sparse_components,
)

DEFAULT_COMPARISON_ROOT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "support_cohesion_component_posterior_bakeoff_960a_smoke_s8_start22"
)

POSTERIOR_MODES: dict[str, dict[str, Any]] = {
    "full": {
        "label": "Full pooled",
        "max_components": None,
        "min_cumulative_weight": 1.0,
    },
    "top1": {
        "label": "Top 1 component",
        "max_components": 1,
        "min_cumulative_weight": 1.0,
    },
    "top2_80": {
        "label": "Top 2 or 80%",
        "max_components": 2,
        "min_cumulative_weight": 0.80,
    },
    "top3_90": {
        "label": "Top 3 or 90%",
        "max_components": 3,
        "min_cumulative_weight": 0.90,
    },
}

PLOT_POLICY_LABELS = {
    "current_start_checked_gap30": "Diverse historical regimes",
    "cohesive_support_gap30": "Nearest similar regimes",
    "cluster_family_support_gap30": "One similar-regime family",
    "kernel_similarity_support_gap30": "Similarity-weighted regimes",
    "start_only_topk": "Same-start control",
}

PLOT_POSTERIOR_LABELS = {
    "full": "All selected regimes",
    "top1": "Strongest regime only",
    "top2_80": "Main-regime view: top 2 / 80%",
    "top3_90": "Main-regime view: top 3 / 90%",
}


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def _terminal_var_es_loss(pnl: np.ndarray, *, alpha: float = 0.95) -> dict[str, float]:
    terminal = np.asarray(pnl, dtype=np.float64)[:, -1]
    losses = -terminal
    var = float(np.nanpercentile(losses, 100.0 * float(alpha)))
    tail = losses[losses >= var]
    return {
        "terminal_var95_loss": var,
        "terminal_expected_shortfall95_loss": (
            float(np.nanmean(tail)) if tail.size else var
        ),
    }


def _energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape((left.shape[0], -1))
    y = np.asarray(right, dtype=np.float64).reshape((right.shape[0], -1))
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    dxy = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1).mean()
    dxx = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1).mean()
    dyy = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1).mean()
    return float(2.0 * dxy - dxx - dyy)


def _select_components(
    components: list[dict[str, Any]],
    *,
    posterior_mode: str,
) -> list[dict[str, Any]]:
    if posterior_mode == "full":
        total = sum(max(float(item.get("weight", 0.0)), 0.0) for item in components)
        selected = [dict(item) for item in components]
        for item in selected:
            item["sparse_weight"] = (
                max(float(item.get("weight", 0.0)), 0.0) / total
                if total > 0.0
                else 1.0 / max(len(selected), 1)
            )
        return selected
    spec = POSTERIOR_MODES[posterior_mode]
    return select_sparse_components(
        components,
        max_components=int(spec["max_components"]),
        min_cumulative_weight=float(spec["min_cumulative_weight"]),
    )


def _case_posterior_result(
    *,
    comparison_root: str | Path,
    case_name: str,
    policy_name: str,
    posterior_mode: str,
) -> dict[str, Any]:
    report_path = _case_report_path(comparison_root, case_name, policy_name)
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = _load_json(report_path)
    arrays_path = report_path.with_name("prefix_latent_story_smoke_arrays.npz")
    if not arrays_path.exists():
        raise FileNotFoundError(arrays_path)
    with np.load(arrays_path, allow_pickle=True) as arrays:
        states = np.asarray(arrays["generated_states"], dtype=np.float64)
        start = np.asarray(arrays["requested_raw"], dtype=np.float64)
        variant = _variant_index(report, int(states.shape[0]))
        variant_states = states[variant]
        variant_start = start[variant]
        components = component_slices_for_variant(
            variant_index=variant,
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
            sample_count=int(variant_states.shape[0]),
        )
    component_rows = []
    for component_no, row in enumerate(components):
        component_rows.append({**row, "component_no": int(component_no)})
    selected = _select_components(component_rows, posterior_mode=posterior_mode)
    sample_indices: list[int] = []
    support_rows = []
    for component in selected:
        start_slice, stop_slice = component["sample_slice"]
        sample_indices.extend(range(int(start_slice), int(stop_slice)))
        support_rows.append(
            {
                "component_no": int(component.get("component_no", 0)),
                "window_index": int(component["window_index"]),
                "weight": float(component.get("weight", 0.0)),
                "posterior_weight": float(component.get("sparse_weight", 0.0)),
                "sample_count": int(component.get("sample_count", 0)),
            }
        )
    selected_states = variant_states[np.asarray(sample_indices, dtype=np.int64)]
    pnl = _portfolio_pnl(selected_states, variant_start)
    return {
        "case": str(case_name),
        "case_label": public_case_label(case_name),
        "policy": str(policy_name),
        "posterior_mode": str(posterior_mode),
        "posterior_label": str(POSTERIOR_MODES[posterior_mode]["label"]),
        "sample_count": int(selected_states.shape[0]),
        "component_count": int(len(support_rows)),
        "support": support_rows,
        "portfolio_stats": _terminal_var_es_loss(pnl),
        "states": selected_states,
        "start": variant_start,
    }


def _support_jaccard(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    a = {int(row["window_index"]) for row in left}
    b = {int(row["window_index"]) for row in right}
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


def _pairwise_case_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    market_indices = [int(FACTOR_INDEX[factor]) for factor in SUMMARY_FACTORS]
    for i, left in enumerate(case_results):
        for right in case_results[i + 1 :]:
            factor_ks = {}
            for factor in SUMMARY_FACTORS:
                idx = int(FACTOR_INDEX[factor])
                factor_ks[factor] = _ks_statistic(
                    left["states"][:, -1, idx],
                    right["states"][:, -1, idx],
                )
            pnl_left = _portfolio_pnl(left["states"], left["start"])
            pnl_right = _portfolio_pnl(right["states"], right["start"])
            rows.append(
                {
                    "left": left["case_label"],
                    "right": right["case_label"],
                    "support_jaccard": _support_jaccard(
                        left["support"], right["support"]
                    ),
                    "mean_factor_terminal_ks": float(np.mean(list(factor_ks.values()))),
                    "factor_terminal_ks": factor_ks,
                    "portfolio_terminal_ks": _ks_statistic(
                        pnl_left[:, -1], pnl_right[:, -1]
                    ),
                    "factor_path_energy": _energy_distance(
                        left["states"][:, :, market_indices],
                        right["states"][:, :, market_indices],
                    ),
                }
            )
    return {
        "pair_count": int(len(rows)),
        "mean_support_jaccard": (
            float(np.mean([row["support_jaccard"] for row in rows])) if rows else 0.0
        ),
        "mean_factor_terminal_ks": (
            float(np.mean([row["mean_factor_terminal_ks"] for row in rows]))
            if rows
            else 0.0
        ),
        "mean_portfolio_terminal_ks": (
            float(np.mean([row["portfolio_terminal_ks"] for row in rows]))
            if rows
            else 0.0
        ),
        "mean_factor_path_energy": (
            float(np.nanmean([row["factor_path_energy"] for row in rows]))
            if rows
            else 0.0
        ),
        "rows": rows,
    }


def build_component_posterior_bakeoff(
    *,
    comparison_root: str | Path,
    cases: tuple[str, ...],
    policies: tuple[str, ...],
    posterior_modes: tuple[str, ...],
) -> dict[str, Any]:
    policy_rows = []
    for policy_name in policies:
        posterior_rows = []
        full_metrics: dict[str, float] | None = None
        for posterior_mode in posterior_modes:
            case_results = [
                _case_posterior_result(
                    comparison_root=comparison_root,
                    case_name=case_name,
                    policy_name=policy_name,
                    posterior_mode=posterior_mode,
                )
                for case_name in cases
            ]
            portfolio_var95 = [
                float(case["portfolio_stats"]["terminal_var95_loss"])
                for case in case_results
            ]
            portfolio_es95 = [
                float(case["portfolio_stats"]["terminal_expected_shortfall95_loss"])
                for case in case_results
            ]
            pairwise = _pairwise_case_metrics(case_results)
            metrics = {
                "mean_factor_terminal_ks": float(pairwise["mean_factor_terminal_ks"]),
                "mean_portfolio_terminal_ks": float(
                    pairwise["mean_portfolio_terminal_ks"]
                ),
                "mean_factor_path_energy": float(pairwise["mean_factor_path_energy"]),
                "portfolio_var95_loss_range": float(
                    np.max(portfolio_var95) - np.min(portfolio_var95)
                ),
                "portfolio_es95_loss_range": float(
                    np.max(portfolio_es95) - np.min(portfolio_es95)
                ),
            }
            if posterior_mode == "full":
                full_metrics = dict(metrics)
            ratio_to_full = {}
            if full_metrics is not None and posterior_mode != "full":
                for key, value in metrics.items():
                    denom = float(full_metrics.get(key, 0.0) or 0.0)
                    ratio_to_full[key] = float(value / denom) if denom > 1e-12 else None
            posterior_rows.append(
                {
                    "posterior_mode": posterior_mode,
                    "posterior_label": str(POSTERIOR_MODES[posterior_mode]["label"]),
                    "case_count": int(len(case_results)),
                    "sample_count_min": int(
                        min(int(case["sample_count"]) for case in case_results)
                    ),
                    "component_count_mean": float(
                        np.mean([int(case["component_count"]) for case in case_results])
                    ),
                    "pairwise": pairwise,
                    "metrics": metrics,
                    "ratio_to_full": ratio_to_full,
                    "cases": [
                        {
                            key: value
                            for key, value in case.items()
                            if key not in {"states", "start"}
                        }
                        for case in case_results
                    ],
                }
            )
        policy_rows.append(
            {
                "policy": policy_name,
                "description": POLICIES[policy_name].description,
                "posterior_modes": posterior_rows,
            }
        )
    return {
        "status": "ok",
        "comparison_root": str(comparison_root),
        "cases": list(cases),
        "policies": policy_rows,
        "posterior_modes": list(posterior_modes),
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Support-Coherence Component-Posterior Bakeoff",
        "",
        "Artifact-only analysis of saved fixed-start rollout decks. Higher KS, "
        "path-energy, and VaR/ES ranges indicate stronger narrative separation; "
        "the start-only null should remain flat.",
        "",
        "| Policy | Posterior | Min samples | Mean components | Support Jaccard | Factor KS | Portfolio KS | Path energy | VaR95 range | ES95 range |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in summary["policies"]:
        for posterior in policy["posterior_modes"]:
            pairwise = posterior["pairwise"]
            metrics = posterior["metrics"]
            lines.append(
                "| {policy} | {posterior} | {samples} | {components:.2f} | "
                "{support:.3f} | {factor:.3f} | {portfolio:.3f} | "
                "{energy:.3f} | {var:.3f} | {es:.3f} |".format(
                    policy=policy["policy"],
                    posterior=posterior["posterior_mode"],
                    samples=int(posterior["sample_count_min"]),
                    components=float(posterior["component_count_mean"]),
                    support=float(pairwise["mean_support_jaccard"]),
                    factor=float(metrics["mean_factor_terminal_ks"]),
                    portfolio=float(metrics["mean_portfolio_terminal_ks"]),
                    energy=float(metrics["mean_factor_path_energy"]),
                    var=float(metrics["portfolio_var95_loss_range"]),
                    es=float(metrics["portfolio_es95_loss_range"]),
                )
            )
    lines.append("")
    lines.append("## Ratios To Full Pooling")
    lines.append("")
    lines.append(
        "| Policy | Posterior | Factor KS ratio | Portfolio KS ratio | Path energy ratio | VaR95 range ratio |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for policy in summary["policies"]:
        for posterior in policy["posterior_modes"]:
            if posterior["posterior_mode"] == "full":
                continue
            ratio = posterior["ratio_to_full"]
            lines.append(
                "| {policy} | {posterior} | {factor} | {portfolio} | {energy} | {var} |".format(
                    policy=policy["policy"],
                    posterior=posterior["posterior_mode"],
                    factor=_fmt_ratio(ratio.get("mean_factor_terminal_ks")),
                    portfolio=_fmt_ratio(ratio.get("mean_portfolio_terminal_ks")),
                    energy=_fmt_ratio(ratio.get("mean_factor_path_energy")),
                    var=_fmt_ratio(ratio.get("portfolio_var95_loss_range")),
                )
            )
    return "\n".join(lines)


def _fmt_ratio(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}x"


def plot_component_posterior_fans(
    *,
    comparison_root: str | Path,
    cases: tuple[str, ...],
    policies: tuple[str, ...],
    posterior_modes: tuple[str, ...],
    output_path: str | Path,
    factors: tuple[str, ...] = SUMMARY_FACTORS,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(policy, mode) for policy in policies for mode in posterior_modes]
    days = np.arange(0, 31)
    fig, axes = plt.subplots(
        len(rows),
        len(factors),
        figsize=(3.7 * len(factors), 2.6 * len(rows)),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(
        "Raw-level factor fans by narrative: all selected regimes versus main-regime view",
        fontsize=14,
        fontweight="bold",
    )
    for row_idx, (policy_name, posterior_mode) in enumerate(rows):
        for col_idx, factor in enumerate(factors):
            ax = axes[row_idx, col_idx]
            idx = int(FACTOR_INDEX[factor])
            for case_name in cases:
                result = _case_posterior_result(
                    comparison_root=comparison_root,
                    case_name=case_name,
                    policy_name=policy_name,
                    posterior_mode=posterior_mode,
                )
                states = np.asarray(result["states"], dtype=np.float64)
                start = float(np.asarray(result["start"], dtype=np.float64)[idx])
                paths = np.concatenate(
                    [
                        np.full((states.shape[0], 1), start, dtype=np.float64),
                        states[:, :, idx],
                    ],
                    axis=1,
                )
                q10, q50, q90 = np.nanpercentile(paths, [10, 50, 90], axis=0)
                color = str(CASE_SPECS[case_name]["color"])
                ax.fill_between(days, q10, q90, color=color, alpha=0.045)
                ax.plot(
                    days,
                    q50,
                    color=color,
                    linewidth=1.25,
                    label=(
                        public_case_label(case_name)
                        if row_idx == 0 and col_idx == 0
                        else None
                    ),
                )
            if row_idx == 0:
                ax.set_title(factor, fontsize=9, fontweight="bold")
            if col_idx == 0:
                policy_label = PLOT_POLICY_LABELS.get(policy_name, policy_name)
                posterior_label = PLOT_POSTERIOR_LABELS.get(
                    posterior_mode,
                    str(POSTERIOR_MODES[posterior_mode]["label"]),
                )
                ax.set_ylabel(
                    f"{policy_label}\n{posterior_label}",
                    fontsize=7,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=58,
                )
            ax.grid(alpha=0.15)
            ax.set_xlim(0, 30)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=3,
        fontsize=8,
        title="Color = narrative; solid line = median; shaded band = 10th-90th percentile",
        title_fontsize=8,
        frameon=False,
    )
    fig.text(
        0.5,
        0.006,
        "Row label format: support selection / scenario view. "
        "Main-regime view keeps the highest-weight historical regimes instead of averaging every selected regime.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#424242",
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.055, 0.105, 1, 0.955])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", default=DEFAULT_COMPARISON_ROOT)
    parser.add_argument("--case", action="append")
    parser.add_argument("--policy", action="append")
    parser.add_argument(
        "--posterior-mode",
        action="append",
        choices=tuple(POSTERIOR_MODES),
    )
    parser.add_argument("--output-json")
    parser.add_argument("--output-markdown")
    parser.add_argument("--plot-output")
    parser.add_argument("--plot-policy", action="append")
    parser.add_argument(
        "--plot-posterior-mode",
        action="append",
        choices=tuple(POSTERIOR_MODES),
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cases = tuple(args.case or CASE_NAMES)
    policies = tuple(args.policy or POLICIES)
    posterior_modes = tuple(args.posterior_mode or POSTERIOR_MODES)
    summary = build_component_posterior_bakeoff(
        comparison_root=args.comparison_root,
        cases=cases,
        policies=policies,
        posterior_modes=posterior_modes,
    )
    output_json = Path(
        args.output_json
        or Path(args.comparison_root) / "component_posterior_bakeoff.json"
    )
    output_markdown = Path(
        args.output_markdown
        or Path(args.comparison_root) / "component_posterior_bakeoff.md"
    )
    plot_output = Path(
        args.plot_output
        or Path(args.comparison_root) / "component_posterior_factor_fans.png"
    )
    summary["artifact_paths"] = {
        "json": str(output_json),
        "markdown": str(output_markdown),
        "factor_fans": None if bool(args.no_plots) else str(plot_output),
    }
    if not bool(args.no_plots):
        plot_component_posterior_fans(
            comparison_root=args.comparison_root,
            cases=cases,
            policies=tuple(args.plot_policy or policies),
            posterior_modes=tuple(args.plot_posterior_mode or ("full", "top3_90")),
            output_path=plot_output,
        )
    _write_json(output_json, summary)
    _write_text(output_markdown, _render_markdown(summary))
    compact = {
        "status": summary["status"],
        "artifact_paths": summary["artifact_paths"],
        "policies": {
            policy["policy"]: {
                posterior["posterior_mode"]: {
                    "factor_ks": posterior["metrics"]["mean_factor_terminal_ks"],
                    "portfolio_ks": posterior["metrics"]["mean_portfolio_terminal_ks"],
                    "var95_range": posterior["metrics"]["portfolio_var95_loss_range"],
                }
                for posterior in policy["posterior_modes"]
            }
            for policy in summary["policies"]
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
