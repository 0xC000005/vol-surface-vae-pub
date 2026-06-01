#!/usr/bin/env python
"""Audit portfolio-level conditionality for narrative scenario casebooks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
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
    empirical_ks,
    empirical_wasserstein,
    load_observed_cases,
)
from experiments.backfill.block_ar.plot_narrative_casebook_backtest import (  # noqa: E402
    CONTROL_ROOT,
    DEFAULT_VARIANT_DIR,
    PORTFOLIO_IMPACT_FIGURE,
    _portfolio_pnl,
)


DEFAULT_CONTROL_ROOT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_component_fixed_decoder_controls_906a_diverse_gap30_s384"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_portfolio_conditionality_audit_906a"
)


def _base_case_name(case_name: str) -> str:
    return str(case_name).split("#seed_")[0]


def _case_pnl(case: dict[str, Any]) -> np.ndarray:
    return _portfolio_pnl(
        np.asarray(case["states"], dtype=np.float32),
        np.asarray(case["start"], dtype=np.float32),
    )


def _terminal_loss_var(pnl: np.ndarray, q: float = 0.05) -> float:
    terminal = np.asarray(pnl, dtype=np.float64)[:, -1]
    return float(-np.quantile(terminal, q))


def _terminal_loss_es(pnl: np.ndarray, q: float = 0.05) -> float:
    terminal = np.asarray(pnl, dtype=np.float64)[:, -1]
    cutoff = np.quantile(terminal, q)
    return float(-np.mean(terminal[terminal <= cutoff]))


def _compare_pnl(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    control: str,
) -> dict[str, Any]:
    left_pnl = _case_pnl(left)
    right_pnl = _case_pnl(right)
    left_terminal = left_pnl[:, -1]
    right_terminal = right_pnl[:, -1]
    return {
        "control": control,
        "left_case": str(left.get("case_name", left.get("label", ""))),
        "right_case": str(right.get("case_name", right.get("label", ""))),
        "path_wasserstein": empirical_wasserstein(left_pnl, right_pnl),
        "terminal_wasserstein": empirical_wasserstein(left_terminal, right_terminal),
        "terminal_ks": empirical_ks(left_terminal, right_terminal),
        "terminal_median_diff": float(np.median(right_terminal) - np.median(left_terminal)),
        "var95_loss_diff": float(_terminal_loss_var(right_pnl) - _terminal_loss_var(left_pnl)),
        "es95_loss_diff": float(_terminal_loss_es(right_pnl) - _terminal_loss_es(left_pnl)),
    }


def observed_pairwise_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _compare_pnl(left, right, control="observed_cross_narrative")
        for left, right in combinations(cases, 2)
    ]


def start_only_pairwise_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _compare_pnl(left, right, control="start_only_null")
        for left, right in combinations(cases, 2)
    ]


def repeat_pairwise_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(_base_case_name(str(case.get("case_name", ""))), []).append(case)
    rows: list[dict[str, Any]] = []
    for group in grouped.values():
        rows.extend(
            _compare_pnl(left, right, control="same_narrative_repeat")
            for left, right in combinations(group, 2)
        )
    return rows


def bootstrap_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        states = np.asarray(case["states"], dtype=np.float32)
        split = states.shape[0] // 2
        if split < 2:
            continue
        left = {
            **case,
            "case_name": f"{case.get('case_name', '')}#bootstrap_a",
            "states": states[:split],
        }
        right = {
            **case,
            "case_name": f"{case.get('case_name', '')}#bootstrap_b",
            "states": states[split : split * 2],
        }
        rows.append(_compare_pnl(left, right, control="within_run_bootstrap"))
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "pair_count": 0,
            "median_path_wasserstein": 0.0,
            "median_terminal_wasserstein": 0.0,
            "median_terminal_ks": 0.0,
            "median_abs_var95_loss_diff": 0.0,
            "median_abs_es95_loss_diff": 0.0,
        }
    return {
        "pair_count": int(len(rows)),
        "median_path_wasserstein": float(median(float(r["path_wasserstein"]) for r in rows)),
        "median_terminal_wasserstein": float(
            median(float(r["terminal_wasserstein"]) for r in rows)
        ),
        "median_terminal_ks": float(median(float(r["terminal_ks"]) for r in rows)),
        "median_abs_var95_loss_diff": float(
            median(abs(float(r["var95_loss_diff"])) for r in rows)
        ),
        "median_abs_es95_loss_diff": float(
            median(abs(float(r["es95_loss_diff"])) for r in rows)
        ),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(float(denominator)) > 1e-12 else float("inf")


def build_portfolio_conditionality_report(
    *,
    observed_cases: list[dict[str, Any]],
    start_only_cases: list[dict[str, Any]],
    repeat_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    observed = observed_pairwise_rows(observed_cases)
    start_only = start_only_pairwise_rows(start_only_cases)
    repeats = repeat_pairwise_rows(repeat_cases)
    bootstraps = bootstrap_rows(observed_cases)
    summaries = {
        "observed_cross_narrative": _summary(observed),
        "start_only_null": _summary(start_only),
        "same_narrative_repeat": _summary(repeats),
        "within_run_bootstrap": _summary(bootstraps),
    }
    observed_summary = summaries["observed_cross_narrative"]
    repeat_summary = summaries["same_narrative_repeat"]
    bootstrap_summary = summaries["within_run_bootstrap"]
    start_summary = summaries["start_only_null"]
    ratios = {
        "path_vs_repeat": _ratio(
            float(observed_summary["median_path_wasserstein"]),
            float(repeat_summary["median_path_wasserstein"]),
        ),
        "path_vs_bootstrap": _ratio(
            float(observed_summary["median_path_wasserstein"]),
            float(bootstrap_summary["median_path_wasserstein"]),
        ),
        "path_vs_start_only": _ratio(
            float(observed_summary["median_path_wasserstein"]),
            float(start_summary["median_path_wasserstein"]),
        ),
        "var95_vs_repeat": _ratio(
            float(observed_summary["median_abs_var95_loss_diff"]),
            float(repeat_summary["median_abs_var95_loss_diff"]),
        ),
        "var95_vs_bootstrap": _ratio(
            float(observed_summary["median_abs_var95_loss_diff"]),
            float(bootstrap_summary["median_abs_var95_loss_diff"]),
        ),
    }
    failures = []
    warnings = []
    if ratios["path_vs_repeat"] < 1.25:
        failures.append("portfolio_path_effect_not_above_repeat")
    if ratios["path_vs_bootstrap"] < 1.0:
        warnings.append("portfolio_path_effect_not_above_bootstrap")
    if ratios["var95_vs_repeat"] < 1.25:
        warnings.append("portfolio_tail_effect_not_above_repeat")
    status = "pass"
    if failures:
        status = "fail"
    elif warnings:
        status = "warning"
    return {
        "scope_note": (
            "Portfolio-level fixed-start conditionality audit. Observed rows "
            "compare different narratives under the same start. Repeat rows "
            "compare the same narrative with different generator seeds. "
            "Bootstrap rows split samples from one run. Start-only rows remove "
            "the narrative channel."
        ),
        "status": status,
        "warnings": warnings,
        "failures": failures,
        "summaries": summaries,
        "ratios": ratios,
        "rows": {
            "observed_cross_narrative": observed,
            "start_only_null": start_only,
            "same_narrative_repeat": repeats,
            "within_run_bootstrap": bootstraps,
        },
    }


def plot_portfolio_conditionality(report: dict[str, Any], output: Path) -> None:
    controls = [
        "observed_cross_narrative",
        "same_narrative_repeat",
        "within_run_bootstrap",
        "start_only_null",
    ]
    labels = ["Observed\nnarratives", "Repeat\nsame story", "Bootstrap\nwithin run", "Start-only\nnull"]
    colors = ["#1565C0", "#6A1B9A", "#EF6C00", "#78909C"]
    path_values = [
        [float(row["path_wasserstein"]) for row in report["rows"].get(control, [])]
        for control in controls
    ]
    var_values = [
        [abs(float(row["var95_loss_diff"])) for row in report["rows"].get(control, [])]
        for control in controls
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle("Portfolio-level fixed-start narrative conditionality controls", fontweight="bold")
    for ax, values, title, ylabel in [
        (axes[0], path_values, "Full-path P&L distribution distance", "Wasserstein distance"),
        (axes[1], var_values, "Terminal 95% loss spread", "Absolute VaR-style loss difference"),
    ]:
        box = ax.boxplot(values, patch_artist=True, tick_labels=labels, showfliers=True)
        for patch, color in zip(box["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.22)
            patch.set_edgecolor(color)
        for line in box["medians"]:
            line.set_color("#263238")
            line.set_linewidth(1.4)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.18)
    status = str(report.get("status", ""))
    ratios = report.get("ratios", {})
    fig.text(
        0.5,
        0.01,
        (
            f"status={status}; path/repeat={float(ratios.get('path_vs_repeat', 0.0)):.2f}; "
            f"path/bootstrap={float(ratios.get('path_vs_bootstrap', 0.0)):.2f}; "
            f"VaR/repeat={float(ratios.get('var95_vs_repeat', 0.0)):.2f}"
        ),
        ha="center",
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-root", type=Path, default=CONTROL_ROOT)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--variant-dir", default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--fan-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path(PORTFOLIO_IMPACT_FIGURE).with_name(
            "narrative_portfolio_conditionality_controls_start18.png"
        ),
    )
    args = parser.parse_args()

    observed = load_observed_cases(
        args.component_root,
        variant_dir=str(args.variant_dir),
        fan_scale=float(args.fan_scale),
    )
    start_only = _load_start_only_cases(args.control_root, fan_scale=float(args.fan_scale))
    repeats = _load_repeat_cases(args.control_root, fan_scale=float(args.fan_scale))
    report = build_portfolio_conditionality_report(
        observed_cases=observed,
        start_only_cases=start_only,
        repeat_cases=repeats,
    )
    output_report = args.output_dir / "portfolio_conditionality_audit.json"
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    plot_portfolio_conditionality(report, args.figure)
    print(
        json.dumps(
            {
                "report": str(output_report),
                "figure": str(args.figure),
                "status": report["status"],
                "ratios": report["ratios"],
                "summaries": report["summaries"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
