#!/usr/bin/env python
"""Summarize fixed-start narrative conditionality evidence.

This report is for the narrative-conditioned scenario-generator paper/demo. It
does not train a model or call OpenAI. It reads the promoted fixed-start
narrative contrast artifacts and turns them into a compact statement of whether
the narrative channel moves generated scenario distributions after the starting
level is held fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CONTRAST_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_control_suite_865a_full_narrative_s192_contrast/"
    "fixed_start_narrative_contrast.json"
)
DEFAULT_CONTROL_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_control_suite_865d_full_s192_symmetric/"
    "fixed_start_control_suite.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_conditionality_report_869a"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _case_family(case_name: str) -> str:
    text = str(case_name)
    marker = "_start"
    if marker in text:
        return text.rsplit(marker, 1)[0]
    return text


def _case_index(case_summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("case_name", "")): row for row in case_summaries}


def _terminal_market_summary(
    case: dict[str, Any],
    markets: list[str],
) -> dict[str, float]:
    terminal = case.get("terminal_summary", {})
    if not isinstance(terminal, dict):
        return {}
    out: dict[str, float] = {}
    for market in markets:
        values = terminal.get(market, {})
        if isinstance(values, dict) and "mean" in values:
            out[market] = _float(values.get("mean"))
    return out


def _top_contrast_rows(
    pairwise: list[dict[str, Any]],
    reliable_starts: set[str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in pairwise
        if str(row.get("start_name", "")) in reliable_starts
    ]
    ranked = sorted(
        eligible,
        key=lambda row: _float(row.get("standardized_l2_gap")),
        reverse=True,
    )
    rows: list[dict[str, Any]] = []
    for row in ranked[: int(limit)]:
        rows.append(
            {
                "start_name": str(row.get("start_name", "")),
                "left_case": str(row.get("left_case", "")),
                "right_case": str(row.get("right_case", "")),
                "left_family": _case_family(str(row.get("left_case", ""))),
                "right_family": _case_family(str(row.get("right_case", ""))),
                "standardized_l2_gap": _float(row.get("standardized_l2_gap")),
                "largest_abs_market_gaps": row.get("largest_abs_market_gaps", []),
            }
        )
    return rows


def build_conditionality_report(
    *,
    contrast_report: str | Path = DEFAULT_CONTRAST_REPORT,
    control_report: str | Path = DEFAULT_CONTROL_REPORT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_pairs: int = 8,
) -> dict[str, Any]:
    contrast = _load_json(contrast_report)
    controls = _load_json(control_report)
    per_start = [
        row
        for row in controls.get("per_start_controls", [])
        if isinstance(row, dict)
    ]
    reliable_starts = {
        str(row.get("start_name", ""))
        for row in per_start
        if str(row.get("status", "")) == "pass"
    }
    observed = controls.get("observed", {}).get("gap_summary", {})
    start_only = controls.get("controls", {}).get("start_only", {}).get(
        "gap_summary", {}
    )
    repeat = controls.get("controls", {}).get("same_narrative_repeat", {}).get(
        "gap_summary", {}
    )
    bootstrap = controls.get("controls", {}).get("within_run_bootstrap", {}).get(
        "gap_summary", {}
    )
    pairwise = [
        row
        for row in contrast.get("pairwise_contrasts", [])
        if isinstance(row, dict)
    ]
    case_summaries = [
        row for row in contrast.get("case_summaries", []) if isinstance(row, dict)
    ]
    cases = _case_index(case_summaries)
    top_rows = _top_contrast_rows(
        pairwise,
        reliable_starts,
        limit=int(top_pairs),
    )
    casebook: list[dict[str, Any]] = []
    for row in top_rows[: min(4, len(top_rows))]:
        markets = [
            str(item.get("market", ""))
            for item in row.get("largest_abs_market_gaps", [])[:5]
            if isinstance(item, dict)
        ]
        left = cases.get(str(row["left_case"]), {})
        right = cases.get(str(row["right_case"]), {})
        casebook.append(
            {
                **row,
                "left_terminal_mean": _terminal_market_summary(left, markets),
                "right_terminal_mean": _terminal_market_summary(right, markets),
                "left_top_support": left.get("top_support", [])[:3],
                "right_top_support": right.get("top_support", [])[:3],
            }
        )

    observed_median = _float(observed.get("overall_median_gap"))
    repeat_median = _float(repeat.get("overall_median_gap"))
    bootstrap_median = _float(bootstrap.get("overall_median_gap"))
    start_only_median = _float(start_only.get("overall_median_gap"))
    supported = (
        observed_median > 0.0
        and start_only_median == 0.0
        and repeat_median < observed_median
        and bootstrap_median < observed_median
        and len(reliable_starts) >= 3
        and _float(contrast.get("start_max_abs_diff")) == 0.0
    )
    output = Path(output_dir)
    report = {
        "status": "pass" if supported else "warning",
        "scope_note": (
            "Post-experiment analysis. This report summarizes existing fixed-start "
            "narrative contrast evidence; it does not call OpenAI or train a model."
        ),
        "inputs": {
            "contrast_report": str(contrast_report),
            "control_report": str(control_report),
        },
        "artifact_paths": {
            "json": str(output / "narrative_conditionality_report.json"),
            "markdown": str(output / "narrative_conditionality_report.md"),
        },
        "headline": {
            "start_max_abs_diff": _float(contrast.get("start_max_abs_diff")),
            "observed_median_gap": observed_median,
            "observed_pair_count": int(observed.get("pair_count", 0) or 0),
            "start_only_median_gap": start_only_median,
            "repeat_median_gap": repeat_median,
            "bootstrap_median_gap": bootstrap_median,
            "repeat_ratio": (
                repeat_median / observed_median if observed_median else None
            ),
            "bootstrap_ratio": (
                bootstrap_median / observed_median if observed_median else None
            ),
            "reliable_start_count": len(reliable_starts),
            "per_start_status_counts": _status_counts(per_start),
        },
        "per_start_controls": per_start,
        "top_pairwise_contrasts": top_rows,
        "recommended_casebook_examples": casebook,
        "decision": (
            "Narrative conditionality is visible above start-only and seed/noise "
            "controls for the reliability-passing starts. Use a compact casebook "
            "for paper/demo explanation; do not claim arbitrary-start production "
            "readiness from this alone."
            if supported
            else "Conditionality evidence is incomplete; run a tighter contrast or "
            "reliability check before paper/demo promotion."
        ),
        "limitations": [
            "Start 178 remains a high-instability hard case in the reliability gate.",
            "The evidence explains distributional movement, not point forecasting.",
            "This is an artifact-level analysis of existing runs, not a new live-user QA set.",
        ],
    }
    _write_json(report["artifact_paths"]["json"], report)
    Path(report["artifact_paths"]["markdown"]).write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    return report


def render_markdown(report: dict[str, Any]) -> str:
    headline = report["headline"]
    lines = [
        "# Narrative Conditionality Report",
        "",
        str(report["scope_note"]),
        "",
        "## Headline",
        "",
        f"- Status: `{report['status']}`",
        f"- Fixed-start max absolute difference: `{headline['start_max_abs_diff']:.6f}`",
        f"- Observed narrative median gap: `{headline['observed_median_gap']:.3f}`",
        f"- Start-only median gap: `{headline['start_only_median_gap']:.3f}`",
        f"- Same-narrative repeat median gap: `{headline['repeat_median_gap']:.3f}`",
        f"- Within-run bootstrap median gap: `{headline['bootstrap_median_gap']:.3f}`",
        f"- Reliable starts: `{headline['reliable_start_count']}`",
        "",
        "## Interpretation",
        "",
        str(report["decision"]),
        "",
        "## Top Narrative Contrasts",
        "",
        "| Start | Left | Right | Std. L2 Gap | Main Markets |",
        "|---|---|---|---:|---|",
    ]
    for row in report["top_pairwise_contrasts"][:8]:
        markets = ", ".join(
            str(item.get("market", ""))
            for item in row.get("largest_abs_market_gaps", [])[:3]
            if isinstance(item, dict)
        )
        lines.append(
            "| "
            f"{row['start_name']} | {row['left_family']} | {row['right_family']} | "
            f"{row['standardized_l2_gap']:.3f} | {markets} |"
        )
    lines.extend(
        [
            "",
            "## Casebook Candidates",
            "",
            "These are good paper/demo examples because the start is fixed and the "
            "narrative contrast moves the generated terminal distribution.",
        ]
    )
    for example in report["recommended_casebook_examples"]:
        lines.extend(
            [
                "",
                f"### {example['left_family']} vs {example['right_family']} at {example['start_name']}",
                "",
                f"- Standardized gap: `{example['standardized_l2_gap']:.3f}`",
                f"- Left terminal means: `{example['left_terminal_mean']}`",
                f"- Right terminal means: `{example['right_terminal_mean']}`",
            ]
        )
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrast-report", default=str(DEFAULT_CONTRAST_REPORT))
    parser.add_argument("--control-report", default=str(DEFAULT_CONTROL_REPORT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-pairs", type=int, default=8)
    args = parser.parse_args()
    report = build_conditionality_report(
        contrast_report=args.contrast_report,
        control_report=args.control_report,
        output_dir=args.output_dir,
        top_pairs=int(args.top_pairs),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "observed_median_gap": report["headline"]["observed_median_gap"],
                "start_only_median_gap": report["headline"]["start_only_median_gap"],
                "repeat_median_gap": report["headline"]["repeat_median_gap"],
                "markdown": report["artifact_paths"]["markdown"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
