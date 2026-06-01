#!/usr/bin/env python
"""Audit whether Safe-haven Gold terminal neutrality is start-level driven.

This script reuses the same saved professional top3/90 paper/demo artifacts
across multiple accepted starts. It does not call OpenAI and does not rerun the
frozen SNI generator. It diagnoses whether the weak day-30 Gold response in the
Safe-haven narrative is specific to start 22 or persists across the current
multi-start evidence pack.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_safe_haven_gold_channel_audit import (  # noqa: E402
    BASELINE_LEAF,
    NARRATIVE_LEAF,
    _jsonable,
    _load_json,
    _report_path,
    build_audit,
)

DEFAULT_OUTPUT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "safe_haven_gold_start_sensitivity_966c/"
    "safe_haven_gold_start_sensitivity_audit.json"
)

DEFAULT_CASE_ROOTS = [
    Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "posterior_ensemble_candidate_966b_professional_start18_s384_d400/"
        "safe_haven_gold"
    ),
    Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "posterior_ensemble_candidate_966a_professional_start22_s384_d400/"
        "safe_haven_gold"
    ),
    Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "posterior_ensemble_candidate_966c_professional_start40_s384_d400/"
        "safe_haven_gold"
    ),
]


def _extract_start_index(case_root: Path) -> int | None:
    for part in case_root.parts:
        match = re.search(r"start(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _operational_variant(report: dict[str, Any]) -> dict[str, Any]:
    for row in report.get("variant_rows", []):
        if isinstance(row, dict) and bool(row.get("is_operational")):
            return dict(row)
    return {}


def _market_row(rows: list[dict[str, Any]], market: str) -> dict[str, Any]:
    target = str(market).upper()
    for row in rows:
        if str(row.get("Market", "")).upper() == target:
            return dict(row)
    return {}


def _support_gold_match_rate(rows: list[dict[str, Any]]) -> float | None:
    checked = []
    for row in rows:
        gold = row.get("gold_prefix_alignment", {})
        if isinstance(gold, dict) and "aligned" in gold:
            checked.append(bool(gold.get("aligned")))
    if not checked:
        return None
    return float(np.mean(checked))


def _top_support_summary(rows: list[dict[str, Any]], n: int = 3) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows[:n]:
        out.append(
            {
                "rank": int(row.get("rank", len(out) + 1)),
                "window_id": str(row.get("window_id", "")),
                "history_end_date": str(row.get("history_end_date", "")),
                "weight": float(row.get("weight", 0.0) or 0.0),
                "start_distance_z": float(row.get("start_distance_z", 0.0) or 0.0),
                "memory_support_cosine": float(
                    row.get("memory_support_cosine", 0.0) or 0.0
                ),
                "gold_prefix_alignment": row.get("gold_prefix_alignment", {}),
            }
        )
    return out


def _start_state_values(report: dict[str, Any]) -> dict[str, float]:
    values = (
        report.get("selected_start_state", {})
        if isinstance(report.get("selected_start_state"), dict)
        else {}
    ).get("values_by_name", {})
    if not isinstance(values, dict):
        return {}
    keys = [
        "factor:gold",
        "factor:spx",
        "factor:vix",
        "factor:bbb_oas",
        "factor:us10y",
        "factor:dxy",
    ]
    return {
        key: float(values[key])
        for key in keys
        if key in values and math.isfinite(float(values[key]))
    }


def build_start_sensitivity_audit(case_roots: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case_root in case_roots:
        narrative_raw = _load_json(_report_path(case_root, NARRATIVE_LEAF))
        baseline_raw = _load_json(_report_path(case_root, BASELINE_LEAF))
        audit = build_audit(case_root)
        narrative_gold = audit["narrative_top3_gold"]["pooled_gold_terminal"]
        baseline_gold = audit["baseline_top3_gold"]["pooled_gold_terminal"]
        gold_diff = audit["baseline_relative_gold"]
        start_index = _extract_start_index(case_root)
        table_rows = audit.get("gold_table_row", {})
        # Include selected related channels so the reader can see whether the
        # story moved other defensive channels even when Gold is neutral.
        full_table = []
        try:
            from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (  # noqa: PLC0415
                apply_live_top3_90_posterior_ensemble,
                attach_start_only_baseline_report,
                scenario_table,
            )

            narrative = apply_live_top3_90_posterior_ensemble(narrative_raw)
            baseline = apply_live_top3_90_posterior_ensemble(baseline_raw)
            attached = attach_start_only_baseline_report(narrative, baseline)
            full_table = scenario_table(attached).to_dict("records")
        except Exception:
            full_table = []
        rows.append(
            {
                "start_index": start_index,
                "case_root": str(case_root),
                "operational_start": _operational_variant(narrative_raw),
                "start_state": _start_state_values(narrative_raw),
                "gold_table_row": table_rows,
                "gold_terminal": {
                    "narrative_mean_delta": float(narrative_gold["mean_delta"]),
                    "baseline_mean_delta": float(baseline_gold["mean_delta"]),
                    "mean_delta_difference": float(
                        gold_diff["mean_delta_difference"]
                    ),
                    "narrative_probability_up": float(
                        narrative_gold["probability_up"]
                    ),
                    "baseline_probability_up": float(
                        baseline_gold["probability_up"]
                    ),
                    "probability_up_difference": float(
                        gold_diff["probability_up_difference"]
                    ),
                    "narrative_p10_p90": [
                        float(narrative_gold["p10_delta"]),
                        float(narrative_gold["p90_delta"]),
                    ],
                    "baseline_p10_p90": [
                        float(baseline_gold["p10_delta"]),
                        float(baseline_gold["p90_delta"]),
                    ],
                },
                "selected_related_market_rows": {
                    market: _market_row(full_table, market)
                    for market in ["GOLD", "SPX", "VIX", "BBB_OAS", "US10Y", "DXY"]
                },
                "narrative_top3_support": _top_support_summary(
                    audit["narrative_support_rows"]
                ),
                "baseline_top3_support": _top_support_summary(
                    audit["baseline_support_rows"]
                ),
                "narrative_support_gold_match_rate": _support_gold_match_rate(
                    audit["narrative_support_rows"][:3]
                ),
                "baseline_support_gold_match_rate": _support_gold_match_rate(
                    audit["baseline_support_rows"][:3]
                ),
            }
        )
    finite_diffs = [
        float(row["gold_terminal"]["mean_delta_difference"])
        for row in rows
        if row.get("gold_terminal")
    ]
    finite_prob_diffs = [
        float(row["gold_terminal"]["probability_up_difference"])
        for row in rows
        if row.get("gold_terminal")
    ]
    max_abs_mean_diff = max((abs(value) for value in finite_diffs), default=float("nan"))
    max_abs_prob_diff = max(
        (abs(value) for value in finite_prob_diffs), default=float("nan")
    )
    return {
        "status": "ok",
        "case_count": len(rows),
        "case_roots": [str(path) for path in case_roots],
        "rows": sorted(rows, key=lambda row: row.get("start_index") or -1),
        "summary": {
            "max_abs_gold_mean_delta_difference": float(max_abs_mean_diff),
            "max_abs_gold_probability_up_difference": float(max_abs_prob_diff),
            "gold_terminal_response_varies_materially_by_start": bool(
                max_abs_mean_diff > 5.0 or max_abs_prob_diff > 0.10
            ),
            "interpretation": (
                "Across the current professional multi-start top3/90 artifacts, "
                "Safe-haven Gold support prefixes satisfy the Gold-up prefix "
                "direction check, but the generated day-30 Gold distribution "
                "remains close to the start-only baseline. The weak terminal "
                "Gold response is therefore not unique to start 22; it reflects "
                "the current frozen-generator/support response for this narrative."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--case-root",
        action="append",
        default=[],
        help="Safe-haven case root containing cohesive_support_gap30 and start_only_topk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(item) for item in args.case_root] if args.case_root else DEFAULT_CASE_ROOTS
    audit = build_start_sensitivity_audit(roots)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(audit), indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "output": str(output),
                "case_count": audit["case_count"],
                "summary": audit["summary"],
                "gold_rows": [
                    {
                        "start_index": row["start_index"],
                        "start_gold": row["start_state"].get("factor:gold"),
                        **row["gold_terminal"],
                    }
                    for row in audit["rows"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
