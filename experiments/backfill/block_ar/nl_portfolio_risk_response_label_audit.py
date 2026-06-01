#!/usr/bin/env python
"""Audit portfolio-risk response labels for narrative-conditioned scenarios.

This diagnostic asks a product-level question: for the same selected start, do
different professional narratives produce different portfolio risk responses
above same-narrative repeat, bootstrap, and start-only controls?
"""

from __future__ import annotations

import argparse
import json
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
    empirical_wasserstein,
    load_observed_cases,
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
    "decoder_component_full906b_diverse_topk_narrative_start_checked_"
    "gap30_temp0p20_s384_gen_temp_0p50"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_risk_response_label_audit_920a"
)

MARKET_INDEX = {
    "SPX": 25,
    "DXY": 28,
    "CRUDE": 31,
    "US10Y": 33,
    "BBB_OAS": 35,
    "GOLD": 37,
    "VIX": 38,
    "IV_ATM_1Y": 17,
}

PORTFOLIO_BOOKS = [
    {
        "name": "equity_beta_carry",
        "label": "Equity beta / carry",
        "purpose": "Long risk assets, short volatility and credit-spread widening.",
        "exposures": {
            "SPX": 1.00,
            "VIX": -0.55,
            "BBB_OAS": -0.50,
            "IV_ATM_1Y": -0.25,
        },
    },
    {
        "name": "credit_duration",
        "label": "Credit + duration",
        "purpose": "Long credit and rates duration, vulnerable to spreads and yield shocks.",
        "exposures": {
            "BBB_OAS": -0.75,
            "US10Y": -0.55,
            "SPX": 0.25,
            "VIX": -0.20,
        },
    },
    {
        "name": "dollar_liquidity",
        "label": "Dollar-liquidity carry",
        "purpose": "Risk/carry book hurt by dollar strength and volatility.",
        "exposures": {
            "DXY": -0.80,
            "SPX": 0.55,
            "BBB_OAS": -0.35,
            "VIX": -0.35,
        },
    },
    {
        "name": "commodity_inflation",
        "label": "Commodity inflation",
        "purpose": "Commodity/inflation-sensitive book with rates vulnerability.",
        "exposures": {
            "CRUDE": 0.75,
            "GOLD": 0.35,
            "US10Y": -0.45,
            "SPX": -0.20,
        },
    },
    {
        "name": "safe_haven_hedge",
        "label": "Safe-haven hedge",
        "purpose": "Defensive book intended to benefit from gold, rates rally, and volatility.",
        "exposures": {
            "GOLD": 0.80,
            "US10Y": -0.45,
            "VIX": 0.45,
            "SPX": -0.35,
        },
    },
    {
        "name": "short_volatility",
        "label": "Short volatility",
        "purpose": "Short-volatility/carry exposure vulnerable to VIX and implied-vol shocks.",
        "exposures": {
            "VIX": -0.75,
            "IV_ATM_1Y": -0.65,
            "SPX": 0.35,
            "BBB_OAS": -0.25,
        },
    },
]


def _base_case_name(case_name: str) -> str:
    return str(case_name).split("#seed_")[0]


def _normalized_market_move(
    states: np.ndarray,
    start: np.ndarray,
    market: str,
) -> np.ndarray:
    idx = int(MARKET_INDEX[market])
    base = float(start[idx])
    scale = max(abs(base), 1.0)
    return (np.asarray(states, dtype=np.float64)[:, :, idx] - base) / scale


def portfolio_pnl(
    states: np.ndarray,
    start: np.ndarray,
    book: dict[str, Any],
) -> np.ndarray:
    result = np.zeros(
        (int(states.shape[0]), int(states.shape[1])),
        dtype=np.float64,
    )
    for market, sensitivity in dict(book["exposures"]).items():
        result += float(sensitivity) * _normalized_market_move(states, start, market)
    return result * 100.0


def _terminal_loss_var(pnl: np.ndarray, q: float = 0.05) -> float:
    terminal = np.asarray(pnl, dtype=np.float64)[:, -1]
    # Signed loss convention: negative means even the 5% tail remains a gain.
    return float(-np.quantile(terminal, q))


def _terminal_loss_es(pnl: np.ndarray, q: float = 0.05) -> float:
    terminal = np.asarray(pnl, dtype=np.float64)[:, -1]
    cutoff = float(np.quantile(terminal, q))
    tail = terminal[terminal <= cutoff]
    if tail.size == 0:
        return float(-cutoff)
    return float(-np.mean(tail))


def _max_path_loss(pnl: np.ndarray) -> np.ndarray:
    return -np.min(np.asarray(pnl, dtype=np.float64), axis=1)


def portfolio_response_label(
    case: dict[str, Any],
    book: dict[str, Any],
) -> dict[str, Any]:
    pnl = portfolio_pnl(
        np.asarray(case["states"], dtype=np.float64),
        np.asarray(case["start"], dtype=np.float64),
        book,
    )
    terminal = pnl[:, -1]
    max_loss = _max_path_loss(pnl)
    return {
        "case_name": str(case.get("case_name", "")),
        "label": str(case.get("label", "")),
        "book": str(book["name"]),
        "book_label": str(book["label"]),
        "terminal_mean": float(np.mean(terminal)),
        "terminal_p10": float(np.quantile(terminal, 0.10)),
        "terminal_p50": float(np.quantile(terminal, 0.50)),
        "terminal_p90": float(np.quantile(terminal, 0.90)),
        "var95_loss": _terminal_loss_var(pnl),
        "es95_loss": _terminal_loss_es(pnl),
        "var95_loss_floor0": float(max(_terminal_loss_var(pnl), 0.0)),
        "es95_loss_floor0": float(max(_terminal_loss_es(pnl), 0.0)),
        "max_path_loss_p50": float(np.quantile(max_loss, 0.50)),
        "max_path_loss_p90": float(np.quantile(max_loss, 0.90)),
        "sample_count": int(pnl.shape[0]),
    }


def _case_pnl(case: dict[str, Any], book: dict[str, Any]) -> np.ndarray:
    return portfolio_pnl(
        np.asarray(case["states"], dtype=np.float64),
        np.asarray(case["start"], dtype=np.float64),
        book,
    )


def _compare_cases(
    left: dict[str, Any],
    right: dict[str, Any],
    book: dict[str, Any],
    *,
    control: str,
) -> dict[str, Any]:
    left_pnl = _case_pnl(left, book)
    right_pnl = _case_pnl(right, book)
    left_terminal = left_pnl[:, -1]
    right_terminal = right_pnl[:, -1]
    return {
        "control": control,
        "book": str(book["name"]),
        "book_label": str(book["label"]),
        "left_case": str(left.get("case_name", "")),
        "right_case": str(right.get("case_name", "")),
        "path_wasserstein": empirical_wasserstein(left_pnl, right_pnl),
        "terminal_wasserstein": empirical_wasserstein(left_terminal, right_terminal),
        "var95_loss_abs_diff": abs(
            _terminal_loss_var(right_pnl) - _terminal_loss_var(left_pnl)
        ),
        "es95_loss_abs_diff": abs(
            _terminal_loss_es(right_pnl) - _terminal_loss_es(left_pnl)
        ),
        "max_path_loss_p90_abs_diff": abs(
            float(np.quantile(_max_path_loss(right_pnl), 0.90))
            - float(np.quantile(_max_path_loss(left_pnl), 0.90))
        ),
    }


def _observed_pairs(
    cases: list[dict[str, Any]],
    book: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _compare_cases(left, right, book, control="observed_cross_narrative")
        for left, right in combinations(cases, 2)
    ]


def _start_only_pairs(
    cases: list[dict[str, Any]],
    book: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _compare_cases(left, right, book, control="start_only_null")
        for left, right in combinations(cases, 2)
    ]


def _repeat_pairs(
    cases: list[dict[str, Any]],
    book: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(_base_case_name(str(case.get("case_name", ""))), []).append(case)
    rows: list[dict[str, Any]] = []
    for group in grouped.values():
        rows.extend(
            _compare_cases(left, right, book, control="same_narrative_repeat")
            for left, right in combinations(group, 2)
        )
    return rows


def _bootstrap_pairs(
    cases: list[dict[str, Any]],
    book: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        states = np.asarray(case["states"], dtype=np.float64)
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
        rows.append(_compare_cases(left, right, book, control="within_run_bootstrap"))
    return rows


def _median_metric(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(median(float(row[key]) for row in rows))


def _ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= 1e-12:
        return float("inf")
    return float(numerator / denominator)


def _book_summary(
    *,
    book: dict[str, Any],
    observed_cases: list[dict[str, Any]],
    repeat_cases: list[dict[str, Any]],
    start_only_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    observed = _observed_pairs(observed_cases, book)
    repeat = _repeat_pairs(repeat_cases, book)
    bootstrap = _bootstrap_pairs(observed_cases, book)
    start_only = _start_only_pairs(start_only_cases, book)
    controls = {
        "observed_cross_narrative": observed,
        "same_narrative_repeat": repeat,
        "within_run_bootstrap": bootstrap,
        "start_only_null": start_only,
    }
    summaries = {
        name: {
            "pair_count": int(len(rows)),
            "median_path_wasserstein": _median_metric(rows, "path_wasserstein"),
            "median_terminal_wasserstein": _median_metric(rows, "terminal_wasserstein"),
            "median_abs_var95_loss_diff": _median_metric(rows, "var95_loss_abs_diff"),
            "median_abs_es95_loss_diff": _median_metric(rows, "es95_loss_abs_diff"),
            "median_abs_max_path_loss_p90_diff": _median_metric(
                rows,
                "max_path_loss_p90_abs_diff",
            ),
        }
        for name, rows in controls.items()
    }
    obs = summaries["observed_cross_narrative"]
    rep = summaries["same_narrative_repeat"]
    boot = summaries["within_run_bootstrap"]
    start = summaries["start_only_null"]
    ratios = {
        "path_vs_repeat": _ratio(
            obs["median_path_wasserstein"],
            rep["median_path_wasserstein"],
        ),
        "path_vs_bootstrap": _ratio(
            obs["median_path_wasserstein"],
            boot["median_path_wasserstein"],
        ),
        "path_vs_start_only": _ratio(
            obs["median_path_wasserstein"],
            start["median_path_wasserstein"],
        ),
        "var95_vs_repeat": _ratio(
            obs["median_abs_var95_loss_diff"],
            rep["median_abs_var95_loss_diff"],
        ),
        "var95_vs_bootstrap": _ratio(
            obs["median_abs_var95_loss_diff"],
            boot["median_abs_var95_loss_diff"],
        ),
        "es95_vs_repeat": _ratio(
            obs["median_abs_es95_loss_diff"],
            rep["median_abs_es95_loss_diff"],
        ),
        "max_loss_p90_vs_repeat": _ratio(
            obs["median_abs_max_path_loss_p90_diff"],
            rep["median_abs_max_path_loss_p90_diff"],
        ),
    }
    warnings = []
    failures = []
    if ratios["path_vs_repeat"] < 1.25:
        failures.append("path_response_not_above_repeat")
    if ratios["path_vs_bootstrap"] < 1.0:
        warnings.append("path_response_not_above_bootstrap")
    if ratios["var95_vs_repeat"] < 1.0:
        warnings.append("var95_response_not_above_repeat")
    if ratios["es95_vs_repeat"] < 1.0:
        warnings.append("es95_response_not_above_repeat")
    status = "fail" if failures else "warning" if warnings else "pass"
    return {
        "book": str(book["name"]),
        "book_label": str(book["label"]),
        "purpose": str(book["purpose"]),
        "exposures": dict(book["exposures"]),
        "status": status,
        "warnings": warnings,
        "failures": failures,
        "summaries": summaries,
        "ratios": ratios,
        "rows": controls,
    }


def _case_response_labels(
    cases: list[dict[str, Any]],
    books: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        portfolio_response_label(case, book)
        for case in cases
        for book in books
    ]


def build_portfolio_risk_response_report(
    *,
    observed_cases: list[dict[str, Any]],
    repeat_cases: list[dict[str, Any]],
    start_only_cases: list[dict[str, Any]],
    books: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    books = list(books or PORTFOLIO_BOOKS)
    book_summaries = [
        _book_summary(
            book=book,
            observed_cases=observed_cases,
            repeat_cases=repeat_cases,
            start_only_cases=start_only_cases,
        )
        for book in books
    ]
    status_counts: dict[str, int] = {}
    for row in book_summaries:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
    pass_count = int(status_counts.get("pass", 0))
    warning_count = int(status_counts.get("warning", 0))
    fail_count = int(status_counts.get("fail", 0))
    if fail_count:
        overall_status = "warning"
        recommendation = "use_portfolio_labels_as_diagnostic_not_support_policy"
    elif warning_count:
        overall_status = "warning"
        recommendation = "candidate_portfolio_labels_need_tail_noise_fix"
    else:
        overall_status = "pass"
        recommendation = "portfolio_labels_ready_for_support_policy_testflight"
    return {
        "scope_note": (
            "Portfolio-risk response-label audit for narrative-conditioned "
            "fixed-start scenarios. Books are normalized exposure vectors, not "
            "priced portfolios. The goal is to test whether narrative changes "
            "are visible in risk-manager-facing VaR/ES/drawdown style outputs "
            "above repeat, bootstrap, and start-only controls. VaR/ES losses are "
            "signed; negative values mean the simulated 5% tail remains a gain, "
            "and floor-zero loss fields are stored in the JSON."
        ),
        "overall_status": overall_status,
        "recommendation": recommendation,
        "status_counts": status_counts,
        "book_count": int(len(books)),
        "case_counts": {
            "observed": int(len(observed_cases)),
            "repeat": int(len(repeat_cases)),
            "start_only": int(len(start_only_cases)),
        },
        "book_summaries": book_summaries,
        "response_labels": _case_response_labels(observed_cases, books),
        "decision": {
            "pass_books": pass_count,
            "warning_books": warning_count,
            "fail_books": fail_count,
            "interpretation": (
                "Portfolio-risk labels are useful if observed cross-narrative "
                "risk responses exceed repeat/bootstrap controls for several "
                "books. If only path metrics pass but VaR/ES remains noisy, the "
                "labels should guide the next support-policy TestFlight but not "
                "replace the simple mixture."
            ),
        },
    }


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Portfolio-Risk Response Label Audit",
        "",
        str(report["scope_note"]),
        "",
        f"Status: `{report['overall_status']}`",
        f"Recommendation: `{report['recommendation']}`",
        f"Status counts: `{json.dumps(report['status_counts'], sort_keys=True)}`",
        "",
        "| Book | Status | Path/Repeat | Path/Bootstrap | VaR95/Repeat | ES95/Repeat | Purpose |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in report["book_summaries"]:
        ratios = row["ratios"]
        lines.append(
            f"| {row['book_label']} | {row['status']} | "
            f"{_format_float(ratios.get('path_vs_repeat'))} | "
            f"{_format_float(ratios.get('path_vs_bootstrap'))} | "
            f"{_format_float(ratios.get('var95_vs_repeat'))} | "
            f"{_format_float(ratios.get('es95_vs_repeat'))} | "
            f"{row['purpose']} |"
        )
    lines.extend(
        [
            "",
            "## Response Labels",
            "",
            "| Narrative | Book | Terminal P50 | Signed VaR95 Loss | Signed ES95 Loss | Max Path Loss P90 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for label in report["response_labels"]:
        lines.append(
            f"| {label['label']} | {label['book_label']} | "
            f"{_format_float(label['terminal_p50'])} | "
            f"{_format_float(label['var95_loss'])} | "
            f"{_format_float(label['es95_loss'])} | "
            f"{_format_float(label['max_path_loss_p90'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_response_heatmap(report: dict[str, Any], output_path: Path) -> None:
    labels = report["response_labels"]
    case_names = list(dict.fromkeys(str(row["label"]) for row in labels))
    book_labels = list(dict.fromkeys(str(row["book_label"]) for row in labels))
    values = np.full((len(case_names), len(book_labels)), np.nan, dtype=np.float64)
    for row in labels:
        i = case_names.index(str(row["label"]))
        j = book_labels.index(str(row["book_label"]))
        values[i, j] = float(row["var95_loss"])
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    im = ax.imshow(values, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(book_labels)), book_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(case_names)), case_names)
    ax.set_title("Narrative portfolio-risk response labels: VaR95 loss")
    ax.set_xlabel("Portfolio risk book")
    ax.set_ylabel("Narrative")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                _format_float(values[i, j]),
                ha="center",
                va="center",
                color="white" if values[i, j] > np.nanmedian(values) else "black",
                fontsize=8,
            )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("VaR95 loss, normalized risk units")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--variant-dir", default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--fan-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    observed_cases = load_observed_cases(
        args.component_root,
        variant_dir=str(args.variant_dir),
        fan_scale=float(args.fan_scale),
    )
    repeat_cases = _load_repeat_cases(args.control_root, fan_scale=float(args.fan_scale))
    start_only_cases = _load_start_only_cases(args.control_root, fan_scale=float(args.fan_scale))
    report = build_portfolio_risk_response_report(
        observed_cases=observed_cases,
        repeat_cases=repeat_cases,
        start_only_cases=start_only_cases,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "portfolio_risk_response_label_audit.json"
    markdown_path = output_dir / "portfolio_risk_response_label_audit.md"
    heatmap_path = output_dir / "portfolio_risk_response_label_heatmap.png"
    report["artifact_paths"] = {
        "report_json": str(json_path),
        "report_markdown": str(markdown_path),
        "var95_heatmap": str(heatmap_path),
    }
    _write_json(json_path, report)
    write_markdown(report, markdown_path)
    plot_response_heatmap(report, heatmap_path)
    print(
        json.dumps(
            {
                "status": report["overall_status"],
                "recommendation": report["recommendation"],
                "status_counts": report["status_counts"],
                "json": str(json_path),
                "markdown": str(markdown_path),
                "heatmap": str(heatmap_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
