#!/usr/bin/env python
"""Product-facing conditionality stress test for NL scenario rollouts.

The fixed-start rollout comparison proves that narrative-conditioned support is
not identical to the start-only null. This script asks the stricter product
question: are the differences visible in the risk channels each narrative is
actually about, and do the portfolio tails move in a way a risk manager can
inspect?

It is offline: it consumes saved component-preserving rollout arrays and writes
plots, a scorecard, and a compact Markdown report. No OpenAI or generator calls
are made.
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

from experiments.backfill.block_ar.nl_fixed_start_rollout_policy_comparison import (  # noqa: E402
    CASE_NAMES,
    FACTOR_INDEX,
    POLICIES,
    PUBLIC_POLICY_LABELS,
    _ks_statistic,
    _portfolio_pnl,
    _support_jaccard,
    load_case_result,
)


DEFAULT_INPUT_ROOT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_fixed_start_policy_comparison_943a_start22_incumbent_clean_s64_d400"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_conditionality_stress_test_943a_start22_incumbent_clean"
)

CASE_LABELS = {
    "fragile_risk_on": "Fragile risk-on",
    "defensive_risk_off": "Defensive risk-off",
    "commodity_inflation": "Commodity inflation",
    "dollar_liquidity": "Dollar liquidity",
    "rates_selloff": "Rates selloff",
    "safe_haven_gold": "Safe-haven gold",
}
CASE_COLORS = {
    "fragile_risk_on": "#1565C0",
    "defensive_risk_off": "#C62828",
    "commodity_inflation": "#EF6C00",
    "dollar_liquidity": "#6A1B9A",
    "rates_selloff": "#00838F",
    "safe_haven_gold": "#2E7D32",
}
CASE_RELEVANT_FACTORS = {
    "fragile_risk_on": ("SPX", "VIX", "BBB_OAS", "IV_ATM_1Y"),
    "defensive_risk_off": ("SPX", "VIX", "BBB_OAS", "IV_ATM_1Y"),
    "commodity_inflation": ("Crude", "US10Y", "SPX", "IV_ATM_1Y"),
    "dollar_liquidity": ("DXY", "BBB_OAS", "VIX", "SPX"),
    "rates_selloff": ("US10Y", "SPX", "DXY", "IV_ATM_1Y"),
    "safe_haven_gold": ("Gold", "US10Y", "VIX", "SPX"),
}
TERMINAL_OVERLAY_FACTORS = (
    "SPX",
    "VIX",
    "BBB_OAS",
    "US10Y",
    "DXY",
    "Crude",
    "Gold",
    "IV_ATM_1Y",
)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.rstrip() + "\n", encoding="utf-8")


def _load_policy_cases(
    *,
    input_root: str | Path,
    policy: str,
    cases: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    return {
        case: load_case_result(
            output_root=input_root,
            case_name=case,
            policy_name=policy,
        )
        for case in cases
    }


def _path_with_start(result: dict[str, Any], factor: str) -> np.ndarray:
    states = np.asarray(result["states"], dtype=np.float64)
    start_value = float(
        np.asarray(result["start"], dtype=np.float64)[FACTOR_INDEX[factor]]
    )
    return np.concatenate(
        [
            np.full((states.shape[0], 1), start_value, dtype=np.float64),
            states[:, :, FACTOR_INDEX[factor]],
        ],
        axis=1,
    )


def _factor_quantiles(
    result: dict[str, Any], factor: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = _path_with_start(result, factor)
    q10, q50, q90 = np.percentile(paths, [10, 50, 90], axis=0)
    return q10, q50, q90


def _normalized_path_matrix(
    result: dict[str, Any], factors: tuple[str, ...]
) -> np.ndarray:
    start = np.asarray(result["start"], dtype=np.float64).reshape(-1)
    cols = []
    for factor in factors:
        idx = FACTOR_INDEX[factor]
        raw = _path_with_start(result, factor)
        scale = max(abs(float(start[idx])), 1.0)
        cols.append((raw - float(start[idx])) / scale)
    return np.concatenate(cols, axis=1)


def _sample_energy_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_samples: int = 192,
    seed: int = 933,
) -> float:
    """Energy distance between two sample clouds, estimated on a fixed subsample."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape == b.shape and np.allclose(a, b):
        return 0.0
    rng = np.random.default_rng(seed)
    if a.shape[0] > max_samples:
        a = a[rng.choice(a.shape[0], size=max_samples, replace=False)]
    if b.shape[0] > max_samples:
        b = b[rng.choice(b.shape[0], size=max_samples, replace=False)]

    def mean_dist(x: np.ndarray, y: np.ndarray) -> float:
        # Chunked pairwise norm keeps memory bounded and avoids scipy dependency.
        total = 0.0
        count = 0
        for start in range(0, x.shape[0], 64):
            chunk = x[start : start + 64]
            dist = np.linalg.norm(chunk[:, None, :] - y[None, :, :], axis=2)
            total += float(np.sum(dist))
            count += int(dist.size)
        return total / max(count, 1)

    return float(2.0 * mean_dist(a, b) - mean_dist(a, a) - mean_dist(b, b))


def _pairwise_stress_metrics(
    cases: dict[str, dict[str, Any]],
    *,
    reference_case: str,
) -> dict[str, Any]:
    rows = []
    case_names = list(cases)
    for i, left_name in enumerate(case_names):
        for right_name in case_names[i + 1 :]:
            left = cases[left_name]
            right = cases[right_name]
            relevant = tuple(
                dict.fromkeys(
                    list(CASE_RELEVANT_FACTORS[left_name])
                    + list(CASE_RELEVANT_FACTORS[right_name])
                )
            )
            factor_ks = {
                factor: _ks_statistic(
                    np.asarray(left["states"])[:, -1, FACTOR_INDEX[factor]],
                    np.asarray(right["states"])[:, -1, FACTOR_INDEX[factor]],
                )
                for factor in relevant
            }
            left_pnl = _portfolio_pnl(left["states"], left["start"])[:, -1]
            right_pnl = _portfolio_pnl(right["states"], right["start"])[:, -1]
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "support_jaccard": _support_jaccard(
                        left["support"], right["support"]
                    ),
                    "relevant_factors": list(relevant),
                    "mean_relevant_terminal_ks": float(
                        np.mean(list(factor_ks.values()))
                    ),
                    "relevant_factor_terminal_ks": factor_ks,
                    "relevant_path_energy": _sample_energy_distance(
                        _normalized_path_matrix(left, relevant),
                        _normalized_path_matrix(right, relevant),
                    ),
                    "portfolio_terminal_ks": _ks_statistic(left_pnl, right_pnl),
                }
            )
    ref_rows = [row for row in rows if reference_case in {row["left"], row["right"]}]
    return {
        "pair_count": int(len(rows)),
        "reference_case": reference_case,
        "mean_support_jaccard": float(np.mean([r["support_jaccard"] for r in rows])),
        "mean_relevant_terminal_ks": float(
            np.mean([r["mean_relevant_terminal_ks"] for r in rows])
        ),
        "mean_relevant_path_energy": float(
            np.mean([r["relevant_path_energy"] for r in rows])
        ),
        "mean_portfolio_terminal_ks": float(
            np.mean([r["portfolio_terminal_ks"] for r in rows])
        ),
        "reference_mean_relevant_terminal_ks": (
            float(np.mean([r["mean_relevant_terminal_ks"] for r in ref_rows]))
            if ref_rows
            else 0.0
        ),
        "reference_mean_relevant_path_energy": (
            float(np.mean([r["relevant_path_energy"] for r in ref_rows]))
            if ref_rows
            else 0.0
        ),
        "rows": rows,
    }


def _policy_scorecard(
    *,
    policy: str,
    cases: dict[str, dict[str, Any]],
    reference_case: str,
) -> dict[str, Any]:
    pairwise = _pairwise_stress_metrics(cases, reference_case=reference_case)
    start_stack = np.stack(
        [np.asarray(case["start"]) for case in cases.values()], axis=0
    )
    direction_statuses = [str(case["direction_status"]) for case in cases.values()]
    var95_losses = [
        float(case["portfolio_stats"]["terminal_var95_loss"]) for case in cases.values()
    ]
    es95_losses = [
        float(case["portfolio_stats"]["terminal_expected_shortfall95_loss"])
        for case in cases.values()
    ]
    support_counts = [len(case["support"]) for case in cases.values()]
    gates = {
        "same_start": float(np.max(np.abs(start_stack - start_stack[0:1]))) <= 1e-7,
        "direction_all_pass": all(status == "pass" for status in direction_statuses),
        "low_support_overlap": pairwise["mean_support_jaccard"] <= 0.25,
        "factor_distribution_response": pairwise["mean_relevant_terminal_ks"] >= 0.10,
        "path_distribution_response": pairwise["mean_relevant_path_energy"] >= 0.01,
        "portfolio_tail_response": pairwise["mean_portfolio_terminal_ks"] >= 0.05
        and (max(var95_losses) - min(var95_losses)) >= 1.0,
    }
    pass_count = int(sum(bool(v) for v in gates.values()))
    status = (
        "pass"
        if pass_count == len(gates)
        else ("warning" if pass_count >= 4 else "fail")
    )
    return {
        "policy": policy,
        "public_label": PUBLIC_POLICY_LABELS.get(policy, policy),
        "status": status,
        "gates": gates,
        "pass_count": pass_count,
        "gate_count": len(gates),
        "support_count_mean": float(np.mean(support_counts)),
        "support_count_min": int(np.min(support_counts)),
        "support_count_max": int(np.max(support_counts)),
        "direction_statuses": direction_statuses,
        "max_abs_start_difference": float(
            np.max(np.abs(start_stack - start_stack[0:1]))
        ),
        "portfolio_var95_loss_range": float(max(var95_losses) - min(var95_losses)),
        "portfolio_es95_loss_range": float(max(es95_losses) - min(es95_losses)),
        "pairwise": pairwise,
    }


def _portfolio_summary(
    cases: dict[str, dict[str, Any]], *, reference_case: str
) -> list[dict[str, Any]]:
    ref = cases[reference_case]
    ref_terminal = _portfolio_pnl(ref["states"], ref["start"])[:, -1]
    ref_var95 = -float(np.percentile(ref_terminal, 5))
    ref_es95 = -float(
        np.mean(ref_terminal[ref_terminal <= np.percentile(ref_terminal, 5)])
    )
    ref_median = float(np.percentile(ref_terminal, 50))
    rows = []
    for case_name, result in cases.items():
        terminal = _portfolio_pnl(result["states"], result["start"])[:, -1]
        q05 = float(np.percentile(terminal, 5))
        var95 = -q05
        es95 = -float(np.mean(terminal[terminal <= q05]))
        median = float(np.percentile(terminal, 50))
        rows.append(
            {
                "case": case_name,
                "label": CASE_LABELS.get(case_name, case_name),
                "terminal_median": median,
                "var95_loss": var95,
                "es95_loss": es95,
                "delta_median_vs_reference": median - ref_median,
                "delta_var95_loss_vs_reference": var95 - ref_var95,
                "delta_es95_loss_vs_reference": es95 - ref_es95,
            }
        )
    return rows


def plot_relevant_factor_panels(
    *,
    cases: dict[str, dict[str, Any]],
    output: str | Path,
    reference_case: str,
) -> None:
    import matplotlib.pyplot as plt

    days = np.arange(31)
    n_rows = len(cases)
    n_cols = max(len(v) for v in CASE_RELEVANT_FACTORS.values())
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 2.35 * n_rows), squeeze=False
    )
    fig.suptitle(
        "Narrative-relevant raw-level fans under the same accepted start",
        fontsize=14,
        fontweight="bold",
    )
    reference = cases[reference_case]
    for row_no, (case_name, result) in enumerate(cases.items()):
        factors = CASE_RELEVANT_FACTORS[case_name]
        color = CASE_COLORS.get(case_name, "#455A64")
        for col_no in range(n_cols):
            ax = axes[row_no, col_no]
            if col_no >= len(factors):
                ax.axis("off")
                continue
            factor = factors[col_no]
            q10, q50, q90 = _factor_quantiles(result, factor)
            ax.fill_between(days, q10, q90, color=color, alpha=0.22)
            ax.plot(days, q50, color=color, linewidth=1.8)
            ax.scatter([0], [q50[0]], color="#111111", s=14, zorder=3)
            if case_name != reference_case:
                _, ref50, _ = _factor_quantiles(reference, factor)
                ax.plot(
                    days,
                    ref50,
                    color="#616161",
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.9,
                )
            if row_no == 0:
                ax.set_title(factor, fontsize=9, fontweight="bold")
            if col_no == 0:
                ax.set_ylabel(CASE_LABELS.get(case_name, case_name), fontsize=8)
            ax.grid(alpha=0.15)
            ax.set_xlim(0, 30)
    axes[-1, 0].set_xlabel("Forward day")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_reference_contrasts(
    *,
    cases: dict[str, dict[str, Any]],
    output: str | Path,
    reference_case: str,
) -> None:
    import matplotlib.pyplot as plt

    contrast_cases = [name for name in cases if name != reference_case]
    n_cols = max(len(CASE_RELEVANT_FACTORS[name]) for name in contrast_cases)
    fig, axes = plt.subplots(
        len(contrast_cases),
        n_cols,
        figsize=(4.2 * n_cols, 2.2 * len(contrast_cases)),
        squeeze=False,
    )
    fig.suptitle(
        f"Raw-level quantile contrast versus {CASE_LABELS[reference_case]}",
        fontsize=14,
        fontweight="bold",
    )
    days = np.arange(31)
    reference = cases[reference_case]
    for row_no, case_name in enumerate(contrast_cases):
        result = cases[case_name]
        color = CASE_COLORS.get(case_name, "#455A64")
        factors = CASE_RELEVANT_FACTORS[case_name]
        for col_no in range(n_cols):
            ax = axes[row_no, col_no]
            if col_no >= len(factors):
                ax.axis("off")
                continue
            factor = factors[col_no]
            q10, q50, q90 = _factor_quantiles(result, factor)
            r10, r50, r90 = _factor_quantiles(reference, factor)
            ax.fill_between(days, q10 - r10, q90 - r90, color=color, alpha=0.18)
            ax.plot(days, q50 - r50, color=color, linewidth=1.6)
            ax.axhline(0.0, color="#757575", linewidth=0.8, linestyle=":")
            if row_no == 0:
                ax.set_title(factor, fontsize=9, fontweight="bold")
            if col_no == 0:
                ax.set_ylabel(CASE_LABELS.get(case_name, case_name), fontsize=8)
            ax.grid(alpha=0.15)
            ax.set_xlim(0, 30)
    axes[-1, 0].set_xlabel("Forward day")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_terminal_overlays(
    *,
    cases: dict[str, dict[str, Any]],
    output: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
    fig.suptitle(
        "Terminal raw-level distribution overlays by narrative",
        fontsize=14,
        fontweight="bold",
    )
    quantile_axis = np.linspace(0.0, 1.0, next(iter(cases.values()))["states"].shape[0])
    for ax, factor in zip(axes.reshape(-1), TERMINAL_OVERLAY_FACTORS, strict=True):
        idx = FACTOR_INDEX[factor]
        for case_name, result in cases.items():
            terminal = np.sort(np.asarray(result["states"])[:, -1, idx])
            ax.plot(
                terminal,
                quantile_axis,
                color=CASE_COLORS.get(case_name, "#455A64"),
                linewidth=1.2,
                label=(
                    CASE_LABELS.get(case_name, case_name)
                    if factor == TERMINAL_OVERLAY_FACTORS[0]
                    else None
                ),
            )
        ax.set_title(factor, fontsize=9, fontweight="bold")
        ax.grid(alpha=0.15)
    axes[0, 0].set_ylabel("Empirical quantile")
    axes[1, 0].set_ylabel("Empirical quantile")
    fig.legend(loc="lower center", ncol=3, fontsize=8, frameon=False)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_portfolio_delta(
    *,
    portfolio_rows: list[dict[str, Any]],
    output: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    rows = [row for row in portfolio_rows if row["case"] != "fragile_risk_on"]
    labels = [row["label"].replace(" ", "\n") for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(
        x - width,
        [row["delta_median_vs_reference"] for row in rows],
        width,
        label="Median P&L",
        color="#546E7A",
    )
    ax.bar(
        x,
        [row["delta_var95_loss_vs_reference"] for row in rows],
        width,
        label="VaR95 loss",
        color="#C62828",
    )
    ax.bar(
        x + width,
        [row["delta_es95_loss_vs_reference"] for row in rows],
        width,
        label="ES95 loss",
        color="#6A1B9A",
    )
    ax.axhline(0.0, color="#757575", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Delta versus fragile risk-on reference")
    ax.set_title(
        "Portfolio tail deltas by narrative under the same start", fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=3)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Narrative Conditionality Stress Test",
        "",
        "This report tests whether fixed-start narrative conditioning is visible in the "
        "risk channels each narrative is about, not only in generic SPX/VIX/Gold fans.",
        "",
        "## Scorecard",
        "",
        "| Policy | Status | Gates | Mean support Jaccard | Relevant KS | Path energy | Portfolio KS | VaR95 range |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["scorecards"]:
        pairwise = row["pairwise"]
        lines.append(
            f"| {row['public_label'].replace(chr(10), ' ')} | {row['status']} | "
            f"{row['pass_count']}/{row['gate_count']} | "
            f"{pairwise['mean_support_jaccard']:.3f} | "
            f"{pairwise['mean_relevant_terminal_ks']:.3f} | "
            f"{pairwise['mean_relevant_path_energy']:.4f} | "
            f"{pairwise['mean_portfolio_terminal_ks']:.3f} | "
            f"{row['portfolio_var95_loss_range']:.3f} |"
        )
    lines.extend(["", "## Product Interpretation", ""])
    best = report["selected_policy_scorecard"]
    lines.append(
        "The selected narrative policy has nonzero support, factor, path, and "
        "portfolio response versus the start-only null. The stress test should be "
        "read as a promotion diagnostic: if future methods improve the support "
        "prior, they should raise the relevant-factor and portfolio-tail columns "
        "without losing start control or provenance."
    )
    lines.append("")
    lines.append(
        f"Selected policy: `{best['policy']}` with status `{best['status']}` "
        f"and gates `{best['pass_count']}/{best['gate_count']}`."
    )
    lines.extend(["", "## Artifacts", ""])
    for key, value in report["artifact_paths"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines)


def build_stress_test(args: argparse.Namespace) -> dict[str, Any]:
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    cases = tuple(args.case or CASE_NAMES)
    policies = tuple(
        args.policy
        or (
            "current_start_checked_gap30",
            "start_only_topk",
        )
    )
    for policy in policies:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy: {policy}")
    selected_policy = str(args.selected_policy)
    if selected_policy not in policies:
        raise ValueError("--selected-policy must be one of --policy values")
    reference_case = str(args.reference_case)
    if reference_case not in cases:
        raise ValueError("--reference-case must be one of --case values")

    loaded = {
        policy: _load_policy_cases(input_root=input_root, policy=policy, cases=cases)
        for policy in policies
    }
    scorecards = [
        _policy_scorecard(
            policy=policy,
            cases=loaded[policy],
            reference_case=reference_case,
        )
        for policy in policies
    ]
    selected_cases = loaded[selected_policy]
    portfolio_rows = _portfolio_summary(selected_cases, reference_case=reference_case)
    artifact_paths = {
        "json": str(output_dir / "conditionality_stress_test.json"),
        "markdown": str(output_dir / "conditionality_stress_test.md"),
        "narrative_relevant_factor_panels": str(
            output_dir / "narrative_relevant_factor_panels.png"
        ),
        "reference_contrast_panels": str(output_dir / "reference_contrast_panels.png"),
        "terminal_distribution_overlays": str(
            output_dir / "terminal_distribution_overlays.png"
        ),
        "portfolio_tail_deltas": str(output_dir / "portfolio_tail_deltas.png"),
    }
    if not bool(args.no_plots):
        plot_relevant_factor_panels(
            cases=selected_cases,
            output=artifact_paths["narrative_relevant_factor_panels"],
            reference_case=reference_case,
        )
        plot_reference_contrasts(
            cases=selected_cases,
            output=artifact_paths["reference_contrast_panels"],
            reference_case=reference_case,
        )
        plot_terminal_overlays(
            cases=selected_cases,
            output=artifact_paths["terminal_distribution_overlays"],
        )
        plot_portfolio_delta(
            portfolio_rows=portfolio_rows,
            output=artifact_paths["portfolio_tail_deltas"],
        )

    report = {
        "status": "ok",
        "scope_note": (
            "Offline fixed-start conditionality stress test. Uses saved "
            "component-preserving rollout arrays; no OpenAI or generator calls."
        ),
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "cases": list(cases),
        "policies": list(policies),
        "selected_policy": selected_policy,
        "reference_case": reference_case,
        "case_relevant_factors": {k: list(v) for k, v in CASE_RELEVANT_FACTORS.items()},
        "scorecards": scorecards,
        "selected_policy_scorecard": next(
            row for row in scorecards if row["policy"] == selected_policy
        ),
        "portfolio_summary": portfolio_rows,
        "artifact_paths": artifact_paths,
    }
    _write_json(artifact_paths["json"], report)
    _write_text(artifact_paths["markdown"], _render_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", action="append")
    parser.add_argument("--policy", action="append")
    parser.add_argument("--selected-policy", default="current_start_checked_gap30")
    parser.add_argument("--reference-case", default="fragile_risk_on")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    report = build_stress_test(args)
    compact = {
        "status": report["status"],
        "output_dir": report["output_dir"],
        "selected_policy": report["selected_policy"],
        "scorecards": [
            {
                "policy": row["policy"],
                "status": row["status"],
                "gates": f"{row['pass_count']}/{row['gate_count']}",
                "mean_support_jaccard": row["pairwise"]["mean_support_jaccard"],
                "mean_relevant_terminal_ks": row["pairwise"][
                    "mean_relevant_terminal_ks"
                ],
                "mean_relevant_path_energy": row["pairwise"][
                    "mean_relevant_path_energy"
                ],
                "mean_portfolio_terminal_ks": row["pairwise"][
                    "mean_portfolio_terminal_ks"
                ],
                "portfolio_var95_loss_range": row["portfolio_var95_loss_range"],
            }
            for row in report["scorecards"]
        ],
        "artifact_paths": report["artifact_paths"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
