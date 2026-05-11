#!/usr/bin/env python
"""Analyze narrative conditionality while holding the starting level fixed."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BAKEOFF_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_narrative_conditionality_856a/"
    "start_conditioned_bakeoff.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_narrative_contrast_856b"
)
DEFAULT_MARKETS = [
    "SPX",
    "VIX",
    "US2Y",
    "US10Y",
    "BBB_OAS",
    "AAA_OAS",
    "DXY",
    "GOLD",
    "CRUDE_OIL",
    "IV_SURFACE",
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


def _summary_std(summary: dict[str, float]) -> float:
    if summary.get("std") is not None:
        return float(summary["std"])
    width = float(summary["p90"]) - float(summary["p10"])
    return width / (2.0 * 1.281551565545)


def _terminal_summary_by_market(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows = report.get("generation", {}).get("terminal_delta_summary", [])
    if not isinstance(rows, list):
        raise ValueError("report generation.terminal_delta_summary is missing")
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("market") is None:
            continue
        market = str(row["market"]).upper()
        output[market] = {
            "mean": float(row.get("mean_terminal_delta", 0.0) or 0.0),
            "p10": float(row.get("p10", 0.0) or 0.0),
            "p50": float(row.get("p50", row.get("mean_terminal_delta", 0.0)) or 0.0),
            "p90": float(row.get("p90", 0.0) or 0.0),
        }
        output[market]["std"] = _summary_std(output[market])
    return output


def _case_distribution(
    row: dict[str, Any],
    *,
    markets: list[str],
) -> dict[str, Any]:
    report_path = Path(str(row["run_report"]))
    report = _load_json(report_path)
    start_values = report["selected_start_state"]["values_by_name"]
    start_state = np.asarray(
        [float(start_values[key]) for key in sorted(start_values)],
        dtype=np.float32,
    )
    terminal_summaries = _terminal_summary_by_market(report)
    market_summaries = {}
    missing = []
    for market in markets:
        if market in terminal_summaries:
            market_summaries[market] = terminal_summaries[market]
        else:
            missing.append(market)
    if missing:
        raise KeyError(f"{report_path}: terminal summaries missing markets {missing}")
    memory_prior = report.get("cached_query", {}).get("memory_prior", {})
    candidate_details = memory_prior.get("candidate_details", [])
    support = []
    if isinstance(candidate_details, list):
        for item in candidate_details[:5]:
            if isinstance(item, dict):
                support.append(
                    {
                        "rank": int(item.get("rank", 0) or 0),
                        "window_index": int(item.get("window_index", -1) or -1),
                        "weight": float(item.get("weight", 0.0) or 0.0),
                        "memory_support_cosine": float(
                            item.get("memory_support_cosine", 0.0) or 0.0
                        ),
                        "history_end_date": str(item.get("history_end_date", "")),
                    }
                )
    return {
        "case_name": str(row.get("case_name", "")),
        "start_name": str(row.get("start_name", "")),
        "run_report": str(report_path),
        "start_state": start_state,
        "terminal_summary": market_summaries,
        "support": support,
        "direction_status": str(row.get("memory_prior_direction_status", "")),
        "support_match_rate": row.get("memory_prior_support_weighted_match_rate"),
        "final_mixture_mismatch_count": int(
            row.get("memory_prior_final_mixture_mismatch_count", 0) or 0
        ),
    }


def _pairwise_contrast(
    cases: list[dict[str, Any]], markets: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(cases, 2):
        market_deltas = []
        squared = 0.0
        for market in markets:
            left_summary = left["terminal_summary"][market]
            right_summary = right["terminal_summary"][market]
            mean_delta = float(left_summary["mean"] - right_summary["mean"])
            pooled = float(
                np.sqrt(
                    (_summary_std(left_summary) ** 2 + _summary_std(right_summary) ** 2)
                    / 2.0
                )
            )
            standardized = mean_delta / pooled if pooled > 1e-12 else 0.0
            squared += standardized * standardized
            market_deltas.append(
                {
                    "market": market,
                    "left_mean_minus_right_mean": mean_delta,
                    "pooled_std": pooled,
                    "standardized_mean_gap": standardized,
                }
            )
        rows.append(
            {
                "start_name": str(left["start_name"]),
                "left_case": left["case_name"],
                "right_case": right["case_name"],
                "standardized_l2_gap": float(np.sqrt(squared)),
                "largest_abs_market_gaps": sorted(
                    market_deltas,
                    key=lambda item: abs(float(item["standardized_mean_gap"])),
                    reverse=True,
                )[:5],
            }
        )
    return sorted(
        rows, key=lambda item: float(item["standardized_l2_gap"]), reverse=True
    )


def _start_block_summaries(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_start: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_start.setdefault(str(case["start_name"]), []).append(case)
    summaries = []
    for start_name, start_cases in sorted(by_start.items()):
        starts = np.stack([case["start_state"] for case in start_cases], axis=0)
        summaries.append(
            {
                "start_name": start_name,
                "case_count": int(len(start_cases)),
                "start_max_abs_diff": float(
                    np.max(np.abs(starts - starts[0][None, :]))
                ),
            }
        )
    return summaries


def build_fixed_start_contrast(
    bakeoff_report: dict[str, Any],
    *,
    markets: list[str],
) -> dict[str, Any]:
    rows = bakeoff_report.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("bakeoff report has no rows")
    cases = [_case_distribution(row, markets=markets) for row in rows]
    start_blocks = _start_block_summaries(cases)
    start_max_abs_diff = max(
        float(block["start_max_abs_diff"]) for block in start_blocks
    )
    pairwise_rows: list[dict[str, Any]] = []
    for block in start_blocks:
        block_cases = [
            case for case in cases if str(case["start_name"]) == block["start_name"]
        ]
        pairwise_rows.extend(_pairwise_contrast(block_cases, markets))
    pairwise_rows = sorted(
        pairwise_rows,
        key=lambda item: float(item["standardized_l2_gap"]),
        reverse=True,
    )
    case_summaries = []
    for case in cases:
        case_summaries.append(
            {
                "case_name": case["case_name"],
                "start_name": case["start_name"],
                "direction_status": case["direction_status"],
                "support_match_rate": case["support_match_rate"],
                "final_mixture_mismatch_count": case["final_mixture_mismatch_count"],
                "terminal_summary": case["terminal_summary"],
                "top_support": case["support"],
                "run_report": case["run_report"],
            }
        )
    return {
        "status": "pass" if start_max_abs_diff <= 1e-6 else "fail",
        "scope_note": (
            "Fixed-start narrative conditionality analysis. The start state is held "
            "constant, so pairwise terminal-distribution gaps are attributable to "
            "narrative-conditioned support/prefix differences, subject to sampling noise."
        ),
        "case_count": int(len(cases)),
        "markets": list(markets),
        "start_max_abs_diff": start_max_abs_diff,
        "start_blocks": start_blocks,
        "case_summaries": case_summaries,
        "pairwise_contrasts": pairwise_rows,
    }


def _format(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fixed-Start Narrative Conditionality Contrast",
        "",
        report["scope_note"],
        "",
        f"- Status: `{report.get('status')}`",
        f"- Case count: `{report.get('case_count')}`",
        f"- Start max absolute difference: `{_format(report.get('start_max_abs_diff'))}`",
        "",
        "## Fixed-Start Blocks",
        "",
        "| Start | Cases | Max Start Diff |",
        "|---|---:|---:|",
    ]
    for block in report.get("start_blocks", []):
        lines.append(
            f"| `{block.get('start_name')}` | `{block.get('case_count')}` | "
            f"`{_format(block.get('start_max_abs_diff'))}` |"
        )
    lines.extend(
        [
            "",
            "## Terminal Mean Deltas",
            "",
            "| Case | Direction | Support Match | Mix Mismatches | "
            + " | ".join(f"{market} mean" for market in report["markets"])
            + " |",
            "|---|---|---:|---:|" + "---:|" * len(report["markets"]),
        ]
    )
    for case in report["case_summaries"]:
        means = [
            _format(case["terminal_summary"][market]["mean"])
            for market in report["markets"]
        ]
        lines.append(
            f"| `{case['case_name']}` | `{case['direction_status']}` | "
            f"`{_format(case['support_match_rate'])}` | "
            f"`{case['final_mixture_mismatch_count']}` | "
            + " | ".join(f"`{value}`" for value in means)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Narrative Gaps",
            "",
            "| Left | Right | Standardized L2 Gap | Largest Market Gaps |",
            "|---|---|---:|---|",
        ]
    )
    for row in report["pairwise_contrasts"]:
        largest = "; ".join(
            f"{item['market']}={float(item['standardized_mean_gap']):.2f}"
            for item in row["largest_abs_market_gaps"]
        )
        lines.append(
            f"| `{row['start_name']} / {row['left_case']}` | `{row['right_case']}` | "
            f"`{_format(row['standardized_l2_gap'])}` | `{largest}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bakeoff-report", type=Path, default=DEFAULT_BAKEOFF_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--markets",
        default=",".join(DEFAULT_MARKETS),
        help="Comma-separated market names to compare.",
    )
    args = parser.parse_args()

    markets = [
        item.strip().upper() for item in str(args.markets).split(",") if item.strip()
    ]
    report = build_fixed_start_contrast(
        _load_json(args.bakeoff_report), markets=markets
    )
    report["inputs"] = {"bakeoff_report": str(args.bakeoff_report)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "fixed_start_narrative_contrast.json"
    markdown_path = args.output_dir / "fixed_start_narrative_contrast.md"
    _write_json(json_path, report)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(json_path),
                "markdown": str(markdown_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
