#!/usr/bin/env python
"""Summarize calibrated live story-deck Gradio sweeps."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MARKETS = [
    "SPX",
    "VIX",
    "US10Y",
    "BBB_OAS",
    "DXY",
    "GOLD",
    "CRUDE_OIL",
]


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _case_support_rows(case: dict[str, Any], report: dict[str, Any]) -> list[dict]:
    summary_rows = [
        row
        for row in _as_list(case.get("support_top_candidates"))
        if isinstance(row, dict)
    ]
    if summary_rows:
        return summary_rows
    memory_prior = _as_dict(_as_dict(report.get("cached_query")).get("memory_prior"))
    return [
        row
        for row in _as_list(memory_prior.get("candidate_details"))
        if isinstance(row, dict)
    ]


def _terminal_summary(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = _as_list(_as_dict(report.get("generation")).get("terminal_delta_summary"))
    return {
        str(row.get("market")): row
        for row in rows
        if isinstance(row, dict) and row.get("market") is not None
    }


def _calibration_metadata(report: dict[str, Any]) -> dict[str, Any]:
    generation = _as_dict(report.get("generation"))
    return _as_dict(generation.get("narrative_ensemble_calibration"))


def build_live_story_deck_analysis(
    summary_path: str | Path,
    *,
    markets: list[str] | None = None,
) -> dict[str, Any]:
    """Build a compact support/calibration/terminal-delta report."""

    source_path = Path(summary_path)
    summary = _load_json(source_path)
    market_names = list(markets or DEFAULT_MARKETS)
    case_rows: list[dict[str, Any]] = []
    support_sets: dict[str, set[str]] = {}

    for case in _as_list(summary.get("cases")):
        if not isinstance(case, dict):
            continue
        snapshot = Path(str(case.get("prefix_report_snapshot_path", "")))
        if not snapshot.exists():
            raise FileNotFoundError(snapshot)
        report = _load_json(snapshot)
        calibration = _calibration_metadata(report)
        support_rows = _case_support_rows(case, report)
        support_windows = [str(row.get("window_id", "")) for row in support_rows]
        support_sets[str(case.get("case_name", ""))] = {
            window for window in support_windows if window
        }
        terminal = _terminal_summary(report)
        case_rows.append(
            {
                "case_name": str(case.get("case_name", "")),
                "status": str(case.get("status", "")),
                "start_index": case.get("start_index"),
                "calibration_applied": bool(calibration.get("applied")),
                "effective_beta": calibration.get("effective_beta"),
                "support_gate": calibration.get("support_gate"),
                "active_direction_count": calibration.get("active_direction_count"),
                "support_count": len(support_rows),
                "support_windows": support_windows,
                "support_weights": [row.get("weight") for row in support_rows],
                "market_implications": _as_list(case.get("market_implications")),
                "terminal_mean_deltas": {
                    market: _as_dict(terminal.get(market)).get("mean_terminal_delta")
                    for market in market_names
                },
                "terminal_p10": {
                    market: _as_dict(terminal.get(market)).get("p10")
                    for market in market_names
                },
                "terminal_p90": {
                    market: _as_dict(terminal.get(market)).get("p90")
                    for market in market_names
                },
                "report_snapshot": str(snapshot),
                "arrays_snapshot": str(case.get("prefix_arrays_snapshot_path", "")),
            }
        )

    pair_rows: list[dict[str, Any]] = []
    for case_a, case_b in itertools.combinations(support_sets, 2):
        support_a = support_sets[case_a]
        support_b = support_sets[case_b]
        union = support_a | support_b
        pair_rows.append(
            {
                "case_a": case_a,
                "case_b": case_b,
                "support_jaccard": (
                    len(support_a & support_b) / len(union) if union else 0.0
                ),
                "shared_support_windows": sorted(support_a & support_b),
            }
        )
    jaccards = [float(row["support_jaccard"]) for row in pair_rows]
    return {
        "status": str(summary.get("status", "")),
        "source_summary": str(source_path),
        "case_count": int(summary.get("case_count", len(case_rows)) or len(case_rows)),
        "pass_count": int(summary.get("pass_count", 0) or 0),
        "fixed_start_index": summary.get("fixed_start_index"),
        "total_openai_tokens": int(summary.get("total_openai_tokens", 0) or 0),
        "calibration_applied_count": sum(
            1 for row in case_rows if bool(row["calibration_applied"])
        ),
        "min_calibration_support_gate": min(
            [
                float(row["support_gate"])
                for row in case_rows
                if row.get("support_gate") is not None
            ],
            default=0.0,
        ),
        "mean_pairwise_support_jaccard": (
            sum(jaccards) / len(jaccards) if jaccards else None
        ),
        "max_pairwise_support_jaccard": max(jaccards) if jaccards else None,
        "pairwise_support_jaccard": pair_rows,
        "cases": case_rows,
        "interpretation": (
            "Fixed-start live story-deck validation checks whether professional "
            "narratives routed through the public Gradio API apply calibration, "
            "preserve per-case evidence, and select distinct support sets. This "
            "is demo-path support conditionality evidence, not a standalone "
            "held-out production promotion."
        ),
    }


def plot_terminal_delta_panel(report: dict[str, Any], output_path: str | Path) -> str:
    """Plot terminal mean deltas by case for the report markets."""

    import matplotlib.pyplot as plt

    cases = _as_list(report.get("cases"))
    if not cases:
        return ""
    first = _as_dict(cases[0])
    markets = list(_as_dict(first.get("terminal_mean_deltas")).keys())
    if not markets:
        return ""
    fig, axes = plt.subplots(
        len(markets), 1, figsize=(10, 1.8 * len(markets)), sharex=True
    )
    if len(markets) == 1:
        axes = [axes]
    labels = [str(_as_dict(case).get("case_name", "")) for case in cases]
    x = range(len(labels))
    for ax, market in zip(axes, markets):
        values = [
            _as_dict(_as_dict(case).get("terminal_mean_deltas")).get(market)
            for case in cases
        ]
        ax.bar(x, [float(value or 0.0) for value in values], color="#2F6B95")
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_ylabel(market)
    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels(labels, rotation=35, ha="right")
    fig.suptitle("Fixed-start live story deck: terminal mean deltas")
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--plot", default="")
    parser.add_argument("--market", action="append", dest="markets", default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    report = build_live_story_deck_analysis(args.summary, markets=args.markets)
    output = (
        Path(args.output)
        if args.output
        else Path(args.summary).with_name("fixed_start_live_story_deck_analysis.json")
    )
    _write_json(output, report)
    plot_path = ""
    if not bool(args.no_plot):
        plot_path = plot_terminal_delta_panel(
            report,
            args.plot
            or str(output.with_name(output.stem + "_terminal_mean_deltas.png")),
        )
    print(
        json.dumps(
            {
                "analysis": str(output),
                "plot": plot_path,
                "case_count": report["case_count"],
                "pass_count": report["pass_count"],
                "calibration_applied_count": report["calibration_applied_count"],
                "max_pairwise_support_jaccard": report["max_pairwise_support_jaccard"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
